import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "results")

# 创建模型目录
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    """
    加载训练数据
    """
    train_data = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    X = train_data.drop('risk_flag', axis=1)
    y = train_data['risk_flag']
    return X, y


def objective(trial, X, y):
    """
    Optuna优化目标函数
    """
    # 定义超参数搜索空间
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42
    }
    
    # 创建模型
    model = XGBClassifier(**params)
    
    # 分割数据用于交叉验证
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        
        # 计算评估指标（使用负MSE作为优化目标）
        mse = mean_squared_error(y_val, y_pred)
        return mse
    except Exception as e:
        # 如果模型训练失败，返回无穷大
        return float('inf')


def train_best_model(X, y, best_params):
    """
    使用最佳参数训练最终模型
    """
    # 创建模型
    model = XGBClassifier(**best_params)
    
    # 训练模型
    model.fit(X, y)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # 打印评估结果
    print("模型性能评估结果：")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mse': mse,
        'mae': mae,
        'r2': r2
    }


def main():
    """
    主函数
    """
    # 加载数据
    print("正在加载数据...")
    X, y = load_data()
    
    # 分割训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用Optuna进行超参数优化
    print("开始超参数优化...")
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, show_progress_bar=True)
    
    # 获取最佳参数
    best_params = study.best_params
    print(f"最佳参数: {best_params}")
    
    # 使用最佳参数训练模型
    print("正在训练最佳模型...")
    best_model = train_best_model(X_train, y_train, best_params)
    
    # 评估模型
    print("正在评估模型...")
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存评估结果
    metrics_path = os.path.join(MODEL_DIR, 'model_metrics.txt')
    with open(metrics_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    print(f"评估结果已保存到: {metrics_path}")


if __name__ == "__main__":
    main()