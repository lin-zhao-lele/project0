# 信用卡用户风险评估模型

本项目使用XGBoost算法构建一个信用卡用户风险评估模型，并通过Optuna进行自动超参数优化。项目包含数据生成、模型训练、模型预测等完整流程。

## 项目结构

```
.
├── generate_data.py         # 数据生成脚本
├── train_model.py           # 模型训练脚本
├── predict.py               # 模型预测脚本
├── README.md                # 项目说明文档
├── results/                 # 数据存储目录
│   ├── train_data.csv       # 训练数据集
│   ├── test_data.csv        # 测试数据集
│   └── inference_data.csv   # 推理数据集
└── models/                  # 模型存储目录
    ├── credit_risk_model.pkl # 训练好的模型
    └── model_metrics.txt     # 模型评估指标
```

## 环境依赖

- Python 3.6+
- scikit-learn
- pandas
- numpy
- optuna
- joblib

安装依赖：

```bash
pip install scikit-learn pandas numpy optuna joblib xgboost
```

## 使用方法

### 1. 生成数据

运行以下命令生成训练、测试和推理数据集：

```bash
python generate_data.py
```

### 2. 训练模型

运行以下命令训练模型并自动调参：

```bash
python train_model.py
```

训练过程中会使用Optuna进行超参数优化，优化的参数包括：

- `C`: 正则化强度，较小的值表示更强的正则化
- `penalty`: 正则化类型 ('l1', 'l2', 'elasticnet')
- `solver`: 优化算法 ('liblinear', 'saga')
- `max_iter`: 最大迭代次数

### 3. 模型预测

运行以下命令使用训练好的模型进行预测：

```bash
python predict.py
```

预测结果将保存在 `results/predictions.csv` 文件中。

如果推理数据包含`risk_flag`标签，脚本还会计算并显示以下性能指标：
- MSE (均方误差)
- MAE (平均绝对误差)
- R2 (决定系数)

## 模型参数说明

### XGBoost 参数

- `n_estimators`: 基学习器的数量
- `max_depth`: 树的最大深度
- `learning_rate`: 学习率
- `subsample`: 训练每棵树时使用的样本比例
- `colsample_bytree`: 训练每棵树时使用的特征比例
- `gamma`: 最小损失减少值
- `reg_alpha`: L1正则化项
- `reg_lambda`: L2正则化项
- `random_state`: 控制随机抽样的伪随机数生成器的种子。

### Optuna 超参数优化

Optuna是一个自动超参数优化框架，通过定义搜索空间和目标函数，自动寻找最优的超参数组合。

在本项目中，Optuna通过最小化验证集上的均方误差(MSE)来寻找最佳的模型参数。

## 模型评估指标

模型训练完成后，会输出以下评估指标：

- **Accuracy**: 准确率，正确预测的样本占总样本的比例
- **Precision**: 精确率，预测为正类的样本中实际为正类的比例
- **Recall**: 召回率，实际为正类的样本中被正确预测为正类的比例
- **F1 Score**: F1分数，精确率和召回率的调和平均数
- **AUC**: ROC曲线下面积，衡量分类器性能的指标
- **MSE**: 均方误差，预测值与真实值之间差异的平方的平均值
- **MAE**: 平均绝对误差，预测值与真实值之间差异的绝对值的平均值
- **R2**: 决定系数，表示模型对目标变量变异性的解释程度

## 数据集说明

生成的数据集包含以下特征：

1. `payment_history_score`: 付款历史分数
2. `credit_utilization_rate`: 信用利用率
3. `debt_to_income_ratio`: 债务收入比
4. `employment_stability`: 就业稳定性
5. `savings_ratio`: 储蓄比率
6. `gender`: 性别
7. `age`: 年龄
8. `marital_status`: 婚姻状况
9. `education_level`: 教育水平
10. `employment_years`: 工作年限
11. `income`: 收入
12. `housing_type`: 住房类型
13. `car_ownership`: 车辆拥有情况
14. `family_size`: 家庭人口数
15. `credit_score`: 信用评分
16. `savings_account`: 储蓄账户
17. `investment_portfolio`: 投资组合
18. `account_balance`: 账户余额
19. `debt_to_income_ratio`: 债务收入比
20. `payment_history`: 付款历史
21. `credit_utilization`: 信用利用率
22. `bankruptcy_history`: 破产历史
23. `inquiry_count`: 查询次数
24. `employment_status`: 就业状况
25. `default_history`: 历史违约次数
26. `years_at_address`: 居住地址年数
27. `loan_count`: 贷款数量
28. `credit_card_count`: 信用卡数量
29. `phone_type`: 手机类型
30. `monthly_spending`: 月支出
31. `internet_usage`: 网络使用情况
32. `social_media_activity`: 社交媒体活动
33. `utility_bills_payment`: 公用事业账单支付
34. `spending_income_ratio`: 月收入比

目标变量：
- `risk_flag`: 风险标识 (0表示低风险，1表示高风险)

## 许可证

本项目为示例项目，仅供学习和参考使用。