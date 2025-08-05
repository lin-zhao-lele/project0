# LSTM股票收盘价预测模型

这个项目使用LSTM神经网络模型来预测股票收盘价。项目包含训练和预测两个主要部分，并使用Optuna进行超参数优化。

## 项目结构

```
.
├── model_args.json     # 模型配置文件
├── train_model.py      # 模型训练脚本
├── predict.py          # 模型预测脚本
├── results/            # 结果输出目录
└── README.md           # 项目说明文件
```

## 配置文件说明

`model_args.json` 包含以下配置项：

- `training`: 训练数据集文件名
- `predict`: 预测数据集文件名
- `predict_length`: 预测长度（暂未使用）
- `auto_tune`: 是否启用超参数优化
- `params`: 模型参数（当`auto_tune`为false时使用）

## 数据集格式

CSV文件应包含以下列：

```
ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,total_share,float_share,free_share,total_mv,circ_mv,adj_factor,adj_close
```

其中：
- `ts_code`: 股票代码
- `trade_date`: 交易日期
- `close`: 收盘价（目标变量）

## 环境要求

- Python 3.7+
- PyTorch 2.2.2
- Optuna
- scikit-learn
- pandas
- numpy

安装依赖：

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
pip install optuna scikit-learn pandas numpy
```

## 使用方法

1. 准备数据：将训练和预测数据集放在 `data/processed/` 目录下

2. 训练模型：

```bash
python train_model.py
```

3. 进行预测：

```bash
python predict.py
```

## 输出文件

所有输出文件都保存在 `results/` 目录下：

- `lstm_model.pth`: 训练好的模型权重
- `train_metrics.json`: 训练集评估指标
- `predict_metrics.json`: 预测集评估指标
- `predictions.csv`: 预测结果，包含交易日期、真实收盘价和预测收盘价
- `deleted_columns.txt`: 训练过程中删除的列信息
- `deleted_columns_predict.txt`: 预测过程中删除的列信息

## 评估指标

模型使用以下指标进行评估：

- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **R2**: 决定系数
- **Accuracy**: 准确率（基于价格变化方向）
- **Precision**: 精确率（基于价格变化方向）
- **Recall**: 召回率（基于价格变化方向）
- **F1 Score**: F1分数（基于价格变化方向）
- **AUC**: ROC曲线下面积（基于价格变化方向）