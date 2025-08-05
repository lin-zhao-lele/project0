# 信用卡用户风险评估 - 随机森林分类模型

本项目使用随机森林分类模型对信用卡用户进行风险评估，支持Optuna自动调参。

## 数据集特征

本项目生成的信用卡用户数据集包含以下特征：

### 基础特征
- `payment_history_score`: 付款历史评分 (0-100)
- `credit_utilization_rate`: 信用利用率 (0.0-1.0)
- `debt_to_income_ratio`: 债务收入比 (0.0-2.0)
- `employment_stability`: 就业稳定性
- `savings_ratio`: 储蓄比率

### 人口统计特征
- `gender`: 性别 (0=女性, 1=男性)
- `age`: 年龄 (18-80)
- `marital_status`: 婚姻状况 (0=未婚, 1=已婚, 2=离异)
- `education_level`: 教育水平 (1=高中以下, 2=高中, 3=专科, 4=本科, 5=研究生及以上)
- `family_size`: 家庭人口数 (1-8)

### 财务特征
- `income`: 年收入 (20,000-200,000)
- `credit_score`: 信用评分 (300-850)
- `account_balance`: 账户余额 (0-50,000)
- `savings_account`: 储蓄账户 (0-100,000)
- `investment_portfolio`: 投资组合 (0-500,000)
- `monthly_spending`: 月支出 (500-15,000)
- `spending_income_ratio`: 月收入比 (0.1-1.0)

### 信用历史特征
- `payment_history`: 付款历史 (0-100)
- `credit_utilization`: 信用利用率 (0.0-1.0)
- `default_history`: 历史违约次数 (0-10)
- `bankruptcy_history`: 破产历史 (0=无, 1=有)
- `inquiry_count`: 查询次数 (0-20)

### 就业与居住特征
- `employment_status`: 就业状况 (0=全职, 1=兼职, 2=自雇, 3=失业)
- `employment_years`: 工作年限 (0-40)
- `years_at_address`: 居住地址年数 (0-50)
- `housing_type`: 住房类型 (0=租房, 1=自有, 2=父母所有)

### 资产特征
- `car_ownership`: 车辆拥有情况 (0=无, 1=有)
- `loan_count`: 贷款数量 (0-15)
- `credit_card_count`: 信用卡数量 (1-10)

### 行为特征
- `phone_type`: 手机类型 (0=预付费, 1=合约, 2=高端)
- `internet_usage`: 网络使用情况 (0=低, 1=中, 2=高)
- `social_media_activity`: 社交媒体活动 (0=低, 1=中, 2=高)
- `utility_bills_payment`: 公用事业账单支付 (0=准时, 1=偶尔延迟, 2=经常延迟)

### 目标变量
- `risk_flag`: 风险标签 (0表示低风险，1表示高风险)

## 模型参数

本项目使用随机森林分类器，通过Optuna进行超参数优化，优化的参数包括：

- `n_estimators`: 树的数量，范围50-300
- `max_depth`: 树的最大深度，范围3-20
- `min_samples_split`: 分割内部节点所需的最小样本数，范围2-20
- `min_samples_leaf`: 叶节点所需的最小样本数，范围1-10
- `max_features`: 寻找最佳分割时考虑的特征数量，可选['sqrt', 'log2', None]

## 性能指标

模型使用以下指标评估性能：

- R2 Score (决定系数): 表示模型对目标变量变异性的解释程度，越接近1表示模型越好
- MSE (均方误差): 表示预测值与实际值之间差异的平方的平均值，越小表示模型越好
- MAE (平均绝对误差): 表示预测值与实际值之间绝对差异的平均值，越小表示模型越好

## 文件说明

- `generate_data.py`: 生成训练和推理用的CSV数据集
- `train_model.py`: 训练模型并保存到本地
- `predict.py`: 使用训练好的模型进行预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 生成数据集:
```bash
python generate_data.py
```

2. 训练模型:
```bash
python train_model.py
```

3. 进行预测:
```bash
python predict.py
```

## 输出文件

所有输出文件都保存在`results`目录下：

- `train_data.csv`: 训练数据集
- `test_data.csv`: 测试数据集
- `inference_data.csv`: 推理数据集（无标签）
- `rf_model.pkl`: 训练好的随机森林模型
- `scaler.pkl`: 特征标准化器
- `predictions.csv`: 测试集的预测结果
- `inference_predictions.csv`: 推理数据的预测结果