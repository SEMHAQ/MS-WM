"""
训练脚本: 训练SSM世界模型及基线模型
"""
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.ssm_world_model import SSMWorldModel
from src.models.baselines import LSTMWorldModel, TransformerWorldModel
from src.data.robot_dataset import create_dataloaders


def load_config(config_path="configs/default.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(model_type, config):
    """构建模型"""
    data_cfg = config["data"]

    if model_type == "ssm":
        cfg = config["ssm_wm"]
        return SSMWorldModel(
            state_dim=data_cfg["state_dim"],
            action_dim=data_cfg["action_dim"],
            d_model=cfg["d_model"],
            d_state=cfg["d_state"],
            n_layers=cfg["n_layers"],
            d_conv=cfg["d_conv"],
            expand=cfg["expand"],
        )
    elif model_type == "lstm":
        cfg = config["lstm_wm"]
        return LSTMWorldModel(
            state_dim=data_cfg["state_dim"],
            action_dim=data_cfg["action_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
        )
    elif model_type == "transformer":
        cfg = config["transformer_wm"]
        return TransformerWorldModel(
            state_dim=data_cfg["state_dim"],
            action_dim=data_cfg["action_dim"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            n_layers=cfg["n_layers"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_epoch(model, loader, optimizer, config, device):
    """训练一个epoch"""
    model.train()
    train_cfg = config["train"]
    losses = defaultdict(float)

    for batch in loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # 单步预测
        pred = model(states, actions)
        loss_single = nn.MSELoss()(pred, target)

        # 多步展开损失 (如果模型支持)
        loss_multi = torch.tensor(0.0, device=device)
        if train_cfg.get("multi_step_loss_weight", 0) > 0:
            H = train_cfg.get("multi_step_horizon", 8)
            if hasattr(model, 'predict_trajectory') and states.shape[1] > H:
                # 使用序列末尾H步作为未来动作
                future_actions = actions[:, -H:]
                init_states = states[:, :-H]
                init_actions = actions[:, :-H]

                pred_traj = model.predict_trajectory(init_states, init_actions, future_actions)
                target_traj = states[:, -H:]
                loss_multi = nn.MSELoss()(pred_traj, target_traj)

        # 总损失
        lambda_multi = train_cfg.get("multi_step_loss_weight", 0.5)
        loss = loss_single + lambda_multi * loss_multi

        loss.backward()
        if train_cfg.get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()

        losses["single"] += loss_single.item()
        losses["multi"] += loss_multi.item()
        losses["total"] += loss.item()

    n = len(loader)
    return {k: v / n for k, v in losses.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    total_mse = 0
    total_mae = 0
    n_batches = 0

    for batch in loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        target = batch["target"].to(device)

        pred = model(states, actions)
        mse = nn.MSELoss()(pred, target).item()
        mae = nn.L1Loss()(pred, target).item()

        total_mse += mse
        total_mae += mae
        n_batches += 1

    return {"mse": total_mse / n_batches, "mae": total_mae / n_batches}


@torch.no_grad()
def measure_inference_time(model, loader, device, n_batches=10):
    """测量推理时间"""
    model.eval()
    times = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        model(states, actions)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # M


def train_model(model_type, config, device):
    """训练指定类型的模型"""
    print(f"\n{'='*60}")
    print(f"  Training: {model_type.upper()} World Model")
    print(f"{'='*60}")

    # 创建数据加载器
    data_cfg = config["data"]
    train_loader, val_loader = create_dataloaders(
        data_dir=data_cfg["data_dir"],
        seq_len=data_cfg["seq_len"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        normalize=data_cfg["normalize"],
    )

    # 构建模型
    model = build_model(model_type, config).to(device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:.2f}M")

    # 优化器和调度器
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"],
    )

    # 训练循环
    best_mse = float('inf')
    save_dir = Path(config["log"]["save_dir"]) / model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(train_cfg["epochs"]):
        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, config, device)
        scheduler.step()

        # 评估
        if (epoch + 1) % config["log"].get("eval_interval", 5) == 0:
            val_metrics = evaluate(model, val_loader, device)

            if val_metrics["mse"] < best_mse:
                best_mse = val_metrics["mse"]
                if config["log"].get("save_best", True):
                    torch.save(model.state_dict(), save_dir / "best.pth")

            print(f"  Epoch {epoch+1:3d}/{train_cfg['epochs']} | "
                  f"Train Loss: {train_losses['total']:.6f} | "
                  f"Val MSE: {val_metrics['mse']:.6f} | "
                  f"Val MAE: {val_metrics['mae']:.6f} | "
                  f"Best: {best_mse:.6f}")

    # 测量推理时间
    avg_time, std_time = measure_inference_time(model, val_loader, device)
    print(f"\n  Results for {model_type.upper()}:")
    print(f"    Best Val MSE: {best_mse:.6f}")
    print(f"    Parameters:   {n_params:.2f}M")
    print(f"    Inference:    {avg_time:.1f} +/- {std_time:.1f} ms")

    return {
        "model": model_type,
        "best_mse": best_mse,
        "params_m": n_params,
        "inference_ms": avg_time,
        "inference_std": std_time,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="all", choices=["ssm", "lstm", "transformer", "all"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # 训练
    if args.model == "all":
        results = []
        for model_type in ["ssm", "lstm", "transformer"]:
            result = train_model(model_type, config, device)
            results.append(result)

        # 打印对比表格
        print(f"\n{'='*70}")
        print(f"  Comparison Results")
        print(f"{'='*70}")
        print(f"  {'Model':<15} {'MSE':<12} {'Params(M)':<12} {'Infer(ms)':<12}")
        print(f"  {'-'*50}")
        for r in results:
            print(f"  {r['model']:<15} {r['best_mse']:<12.6f} {r['params_m']:<12.2f} {r['inference_ms']:<12.1f}")
    else:
        train_model(args.model, config, device)


if __name__ == "__main__":
    main()
