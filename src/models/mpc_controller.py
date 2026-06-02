"""
基于SSM世界模型的模型预测控制(MPC)
"""
import torch
import torch.nn as nn
import numpy as np


class MPCController:
    """
    模型预测控制器

    使用训练好的世界模型进行在线优化控制:
    1. 在每个控制时刻, 优化未来H步的动作序列
    2. 执行第一个动作
    3. 观测新状态, 重复步骤1
    """

    def __init__(
        self,
        world_model: nn.Module,
        horizon: int = 10,
        Q_weight: float = 1.0,
        R_weight: float = 0.01,
        n_iterations: int = 50,
        lr: float = 0.01,
        device: str = "cpu",
    ):
        self.world_model = world_model
        self.horizon = horizon
        self.Q_weight = Q_weight
        self.R_weight = R_weight
        self.n_iterations = n_iterations
        self.lr = lr
        self.device = torch.device(device)

        self.world_model.eval()
        self.world_model.to(self.device)

    @torch.no_grad()
    def plan(self, state_history, action_history, target_state):
        """
        规划: 给定历史状态和动作, 计算最优动作序列

        Args:
            state_history:  (T, state_dim) 历史状态
            action_history: (T-1, action_dim) 历史动作
            target_state:   (state_dim,) 目标状态

        Returns:
            best_action: (action_dim,) 最优的第一步动作
        """
        T = state_history.shape[0]
        action_dim = action_history.shape[1]

        # 初始化动作序列 (可学习参数)
        action_sequence = nn.Parameter(
            torch.randn(self.horizon, action_dim, device=self.device) * 0.1
        )
        optimizer = torch.optim.Adam([action_sequence], lr=self.lr)

        state_history = torch.FloatTensor(state_history).unsqueeze(0).to(self.device)
        action_history = torch.FloatTensor(action_history).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target_state).to(self.device)

        for _ in range(self.n_iterations):
            optimizer.zero_grad()

            # 构建完整输入序列
            full_states = state_history.clone()
            full_actions = action_history.clone()

            total_cost = torch.tensor(0.0, device=self.device, requires_grad=False)
            for h in range(self.horizon):
                pred = self.world_model(full_states, full_actions)

                # 状态跟踪代价
                state_cost = self.Q_weight * torch.norm(pred - target, p=2)

                # 动作正则化代价
                action_cost = self.R_weight * torch.norm(action_sequence[h], p=2)

                total_cost = total_cost + state_cost + action_cost

                # 更新序列
                full_states = torch.cat([full_states[:, 1:], pred.unsqueeze(1)], dim=1)
                full_actions = torch.cat([full_actions[:, 1:], action_sequence[h:h+1].unsqueeze(0)], dim=1)

            total_cost.backward()
            optimizer.step()

        return action_sequence[0].detach().cpu().numpy()

    def closed_loop_control(
        self,
        init_state,
        init_actions,
        target_state,
        true_dynamics=None,
        n_steps=50,
    ):
        """
        闭环控制仿真

        Args:
            init_state:    (T, state_dim) 初始状态序列
            init_actions:  (T-1, action_dim) 初始动作序列
            target_state:  (state_dim,) 目标状态
            true_dynamics: 真实动力学函数 (用于仿真)
            n_steps:       控制步数

        Returns:
            trajectory: (n_steps, state_dim) 状态轨迹
            actions:    (n_steps, action_dim) 执行的动作
        """
        state_history = init_state.copy()
        action_history = init_actions.copy()

        trajectory = [state_history[-1]]
        executed_actions = []

        for step in range(n_steps):
            # 规划
            best_action = self.plan(state_history, action_history, target_state)
            executed_actions.append(best_action)

            # 执行 (如果有真实动力学)
            if true_dynamics is not None:
                new_state = true_dynamics(state_history[-1], best_action)
            else:
                # 使用世界模型自身作为动力学 (开环)
                with torch.no_grad():
                    s = torch.FloatTensor(state_history).unsqueeze(0).to(self.device)
                    a = torch.FloatTensor(action_history).unsqueeze(0).to(self.device)
                    new_state = self.world_model(s, a).cpu().numpy().flatten()

            # 更新历史
            state_history = np.vstack([state_history[1:], new_state.reshape(1, -1)])
            action_history = np.vstack([action_history[1:], best_action.reshape(1, -1)])
            trajectory.append(new_state)

        return np.array(trajectory), np.array(executed_actions)
