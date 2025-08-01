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
