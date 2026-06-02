#!/bin/bash
# 快速启动脚本

# 安装依赖
if [ "$1" = "install" ]; then
    pip install -r requirements.txt
    exit 0
fi

# 训练所有模型
if [ "$1" = "train" ]; then
    python -m src.train.train --config configs/default.yaml --model ${2:-all}
    exit 0
fi

# 训练单个模型
if [ "$1" = "train-ssm" ]; then
    python -m src.train.train --config configs/default.yaml --model ssm
    exit 0
fi

if [ "$1" = "train-lstm" ]; then
    python -m src.train.train --config configs/default.yaml --model lstm
    exit 0
fi

if [ "$1" = "train-transformer" ]; then
    python -m src.train.train --config configs/default.yaml --model transformer
    exit 0
fi

# 快速测试
if [ "$1" = "test" ]; then
    python scripts/quick_test.py
    exit 0
fi

echo "Usage:"
echo "  bash run.sh install          # 安装依赖"
echo "  bash run.sh train            # 训练所有模型"
echo "  bash run.sh train-ssm        # 仅训练SSM模型"
echo "  bash run.sh train-lstm       # 仅训练LSTM模型"
echo "  bash run.sh train-transformer # 仅训练Transformer模型"
echo "  bash run.sh test             # 快速测试"
