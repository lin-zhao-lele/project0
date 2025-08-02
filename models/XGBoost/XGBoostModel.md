# 配置文件 XGBoostModel_args.json

```
"auto_tune": true → 自动调参   false → 用配置里的参数直接训练

"model":     "true 表示只加载已有模型 FinalModel 进行预测，false 表示重新训练并保存模型"

偏稳的版本（效果更准但训练更久）
"params": {
    "n_estimators": 800,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}


```

# XGBoostModelSlidingWindow

``` 
{
  "training": "600519.SH_20150101_20241231_1day_A.csv",
  "predict": "600519.SH_20250101_20250730_1day_A.csv",
  "model": false,
  "auto_tune": true,
  "params": {
    "window_size": 5,
    "n_estimators": 323,
    "max_depth": 10,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "gamma": 0,
    "random_state": 42,
    "n_jobs": -1
  }
}



✅ 参数说明：
参数名	含义
window_size	滑动窗口大小（几天作为一个输入样本）
n_estimators	基学习器数量（树的数量）
max_depth	每棵树的最大深度
learning_rate	学习率
subsample	每棵树使用的训练样本比例
colsample_bytree	每棵树使用的特征比例
min_child_weight	最小样本权重（控制过拟合）
gamma	最小损失下降值（用于剪枝，越大越保守）
random_state	随机种子
n_jobs	多线程（-1 表示用所有核心）

```