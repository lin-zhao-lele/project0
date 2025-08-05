# 机器学习线性回归房价预测模型

这个项目实现了一个基于线性回归的房价预测模型，支持自动调参，并提供了完整的训练和评估流程。

## 项目结构

- `requirements.txt`: 项目依赖
- `generate_data.py`: 生成房价预测的训练和测试数据
- `train_model.py`: 训练和评估线性回归模型
- `train_data.csv`: 训练数据集（运行generate_data.py后生成）
- `test_data.csv`: 测试数据集（运行generate_data.py后生成）
- `model_comparison.png`: 模型性能比较图（运行train_model.py后生成）
- `prediction_results.png`: 预测结果图（运行train_model.py后生成）
- `prediction_results.csv`: 预测结果数据（运行train_model.py后生成）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 生成数据集：

```bash
python generate_data.py
```

这将生成 `train_data.csv` 和 `test_data.csv` 两个文件。

2. 训练和评估模型：

```bash
python train_model.py
```

这将：
- 使用多种线性回归模型（LinearRegression, Ridge, Lasso, ElasticNet）进行训练
- 对需要调参的模型进行自动调参
- 输出各个模型的性能指标（R2, MSE, MAE）
- 生成模型性能比较图 `model_comparison.png`
- 使用最佳模型进行预测并生成预测结果图 `prediction_results.png`
- 保存预测结果到 `prediction_results.csv`

## 模型指标说明

- **R2 (决定系数)**: 表示模型解释目标变量方差的比例，取值范围[0,1]，越接近1表示模型拟合效果越好。
- **MSE (均方误差)**: 预测值与真实值之间差异的平方的平均值，值越小表示模型性能越好。
- **MAE (平均绝对误差)**: 预测值与真实值之间差异的绝对值的平均值，值越小表示模型性能越好。