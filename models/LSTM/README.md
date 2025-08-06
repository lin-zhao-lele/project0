# LSTM股票价格预测模型

## 项目概述
本项目使用PyTorch实现了一个简单的LSTM模型，用于股票收盘价的预测。模型采用滑动窗口技术进行单步预测，并使用Optuna进行超参数优化。

## 文件结构
```
LSTM/
├── processed/                # 存放数据文件
│   ├── 600519.SH_20150101_20241231_1day_A.csv  # 训练数据
│   └── 600519.SH_20250101_20250730_1day_A.csv  # 预测数据
├── results/                  # 存放结果文件
├── train_model.py            # 训练脚本
├── predict.py                # 预测脚本
├── model_args.json           # 配置文件
└── README.md                 # 说明文件
```

## 使用方法

### 1. 准备数据
将训练数据和预测数据放入`processed`目录，并在`model_args.json`中配置文件名。

### 2. 训练模型
运行训练脚本：
```bash
python train_model.py
```
训练过程将自动进行超参数优化，并记录训练时间和性能指标到`results/performance.log`。

### 3. 进行预测
运行预测脚本：
```bash
python predict.py
```
预测结果将保存到`results/predictions.csv`，包含日期、真实值和预测值。

## 技术细节
- **滑动窗口**: 窗口大小30天
- **数据归一化**: 使用MinMaxScaler
- **设备选择**: 自动检测CUDA
- **超参数优化**: 使用Optuna
- **评估指标**: 均方误差(MSE)

## 注意事项
1. 预测数据的前30天不会产生预测结果（用于构建初始窗口）
2. 确保`processed`目录中存在指定的数据文件
3. 所有结果文件将保存在`results`目录