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



## 自动生成Spec

````
参考经典的LSTM模型代码，使用python构建LSTM模型，用于股票收盘价的预测。构造单独的train_model.py  
predict.py  配置 model_args.json ，生成中文readme.md。

目标是生成一个精简但是绝对正确的LSTM模型，用于学习LSTM模型的原理。

每个python文件都包含下面代码做路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__)).
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)).
DATA_DIR = os.path.join(os.path.join(BASE_DIR, "processed").
RESULTS_DIR = os.path.join(BASE_DIR, "results").

model_args.json 的结构如下
{
    "training": "600519.SH_20150101_20241231_daily_adjusted.csv",
    "predict": "600519.SH_20250101_20250730_daily_adjusted.csv",
    "debugLog": false
    "window_size": 30,
    "best_params": {
    }
}

csv文件头为[ ts_code	trade_date  open high low	 close 等等...... ]
其中ts_code为股票代码, 不要当做特征处理应该排除.  trade_date为日期，目标变量：“close”；
"training" 是模型训练数据集 "predict"是推理数据集，
"training" 是模型训练数据集， "training"里面前80%数据用于训练模型，
"training"里面后20%用做验证集;

使用滑动窗口技术，"window_size"为模型训练阶段获得的最佳窗口大小，
"best_params"是其他最佳参数；模型训练结束后将最佳模型的"window_size"和"best_params"
写回model_args.json ；将最佳模型和缩放器都保存起来，方便predict.py在推理阶段加载和使用。
predict.py通过model_args.json读取最佳模型的"window_size"和"best_params"；
确保推理阶段使用和训练阶段相同的模型结构，缩放器和最佳参数；
"debugLog"是代码运行阶段是否输出debugLog，如何为true则将debug信息写入debug.log文件；
构建代码的时候选择关键信息，作为debug Log输出，受"debugLog"参数控制；


设备选择 torch.device("cuda" if torch.cuda.is_available() else "cpu")；

train_model.py 记录训练模型（包括调参阶段）所用的时间，记入performance.log；

主要功能要求：
使用 pytorch框架， Version: 2.2.2；
所有产生的文件都放入RESULTS_DIR；
滑动窗口（window_size = 30）；
单步预测（One-step Ahead）, 模型每次只预测未来一个时间步。；
Optuna自动超参数优化；
数据归一化（保存缩放器对象，推理时加载），目标是能够做到推理结束后能正确还原股价；
训练代码里需要对 目标close 做标准化，推理阶段做相同的标准化，最终保存的预测结果要做正确的反归一化；
输出评估指标到 console 和 performance.log ，内容不要中文，包含训练和推理阶段；
推理结果保存为CSV, 将 "trade_date" ，真实值 "true_close"， 预测值"predicted_close" 保存
到csv文件。注意保存的"true_close" 和"predicted_close"都是真实的股价，即反归一化后的股价；



```



