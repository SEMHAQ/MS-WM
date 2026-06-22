"""
更新论文表格 - 从实验结果自动生成LaTeX表格
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'experiments' / 'paper'
PAPER_DIR = ROOT / 'paper'


def load_results(filename):
    """加载实验结果"""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"错误: {filepath} 不存在")
        return None
    with open(filepath) as f:
        return json.load(f)


def format_mse(mean, std):
    """格式化MSE为LaTeX格式"""
    return f"{mean*100:.2f}$\\pm${std*100:.2f}"


def generate_main_table(results, dataset_name):
    """生成主实验表格"""
    models = ['LSTM-WM', 'GRU-WM', 'Transformer-WM', 'Mamba-WM', 'S4D-WM', 'FSM-WM']

    lines = []
    lines.append("\\begin{tabular}{@{}lcccc@{ }}\\toprule")
    lines.append("方法 & MSE$^{\\text{c}}$ & $R^2$ & 时间$^{\\text{a}}$ & 参数$^{\\text{b}}$ \\\\")
    lines.append("\\midrule")

    for model in models:
        key = f"{model}_{dataset_name}"
        if key not in results:
            continue

        r = results[key]
        mse = format_mse(r['mse_mean'], r['mse_std'])

        # 获取第一个seed的详细信息
        first_seed = list(r['seeds'].values())[0]
        r2_mean = first_seed.get('r2', 0)  # 需要计算R²的均值
        infer_ms = first_seed.get('infer_ms', 0)
        params = first_seed.get('params_m', 0)

        # 计算R²的均值和标准差
        r2_values = [s.get('r2', 0) for s in r['seeds'].values()]
        r2_mean = sum(r2_values) / len(r2_values)
        r2_std = (sum((x - r2_mean)**2 for x in r2_values) / len(r2_values)) ** 0.5

        if model == 'FSM-WM':
            lines.append(f"\\textbf{{{model}}} & \\textbf{{{mse}}} & \\textbf{{{r2_mean:.3f}$\\pm${r2_std:.3f}}} & \\textbf{{{infer_ms:.1f}}} & \\textbf{{{params:.2f}}} \\\\")
        else:
            lines.append(f"{model} & {mse} & {r2_mean:.3f}$\\pm${r2_std:.3f} & {infer_ms:.1f} & {params:.2f} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("更新论文表格")
    print("=" * 60)

    # 加载主实验结果
    main_results = load_results('main_results.json')
    if main_results is None:
        print("请先运行实验: python scripts/run_all_experiments.py")
        return

    # 生成表格
    print("\n=== Humanoid 表格 ===")
    print(generate_main_table(main_results, 'humanoid'))

    print("\n=== Ant 表格 ===")
    print(generate_main_table(main_results, 'ant'))

    print("\n=== Hopper 表格 ===")
    print(generate_main_table(main_results, 'hopper'))

    # 打印摘要
    print("\n" + "=" * 60)
    print("摘要:")
    print("=" * 60)

    for dataset in ['humanoid', 'ant', 'hopper']:
        print(f"\n{dataset}:")
        for model in ['FSM-WM', 'S4D-WM', 'Mamba-WM', 'Transformer-WM']:
            key = f"{model}_{dataset}"
            if key in main_results:
                r = main_results[key]
                print(f"  {model:16s} MSE={r['mse_mean']*100:.2f}±{r['mse_std']*100:.2f}")

    print(f"\n结果保存在: {RESULTS_DIR}")
    print("请手动更新 paper/main.tex 中的表格")


if __name__ == '__main__':
    main()
