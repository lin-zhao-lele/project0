# LSTM股票收盘价预测模型

这是一个使用PyTorch框架构建的LSTM神经网络模型，专门用于股票收盘价的预测。该项目采用滑动窗口技术和Optuna自动超参数优化，实现了精确的单步预测功能。

## 项目结构

```
project/
├── model_args.json          # 配置文件
├── train_model.py          # 模型训练脚本
├── predict.py              # 模型预测脚本
├── processed/              # 数据目录
│   ├── 600519.SH_20150101_20241231_daily_adjusted.csv  # 训练数据
│   └── 600519.SH_20250101_20250730_daily_adjusted.csv  # 预测数据
└── results/                # 结果输出目录
    ├── best_model.pth      # 最佳模型文件
    ├── scalers.pkl         # 数据缩放器
    ├── predictions.csv     # 预测结果
    ├── performance.log     # 性能日志
    └── debug.log          # 调试日志（可选）
```

## 环境要求

- Python 3.8+
- PyTorch 2.2.2
- pandas
- numpy
- scikit-learn
- optuna
- pickle

## 安装依赖

```bash
pip install torch==2.2.2 pandas numpy scikit-learn optuna
```

## 数据格式

CSV文件应包含以下列：
- `ts_code`: 股票代码（训练时会被排除）
- `trade_date`: 交易日期
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价（目标变量）
- 其他数值特征列

## 配置说明

`model_args.json` 配置文件包含以下参数：

- `training`: 训练数据文件名
- `predict`: 预测数据文件名
- `debugLog`: 是否启用调试日志（true/false）
- `window_size`: 滑动窗口大小（默认30）
- `best_params`: 最佳超参数（训练后自动填充）

## 使用方法

### 1. 模型训练

```bash
python train_model.py
```

训练过程包括：
- 数据加载和预处理
- 自动特征标