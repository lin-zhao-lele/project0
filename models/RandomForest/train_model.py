import pandas as pd
import numpy as np
import optuna
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)


def load_data():
    """
    加载训练数据
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'train_data.csv'))
    X = data.drop('risk_flag', axis=1)
    y = data['risk_flag']
    return X, y


def objective(trial, X, y):
    """
    Optuna优化目标函数
    """
    # 定义超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    # 创建模型
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    
    # 分割数据
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_val)
    
    # 计算均方误差作为优化目标
    mse = mean_squared_error(y_val, y_pred)
    
    return mse


def train_model():
    """
    训练随机森林模型
    """
    # 加载数据
    X, y = load_data()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_scaled, y), n_trials=50)
    
    # 输出最佳参数
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 使用最佳参数训练最终模型
    best_rf = RandomForestClassifier(**study.best_params, random_state=42)
    best_rf.fit(X_scaled, y)
    
    # 保存模型和标准化器
    joblib.dump(best_rf, os.path.join(DATA_DIR, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))
    
    # 在训练集上评估模型
    y_pred = best_rf.predict(X_scaled)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print("\n模型性能指标:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    return best_rf, scaler


if __name__ == "__main__":
    model, scaler = train_model()
    print("\n模型已保存到 rf_model.pkl")
    print("标准化器已保存到 scaler.pkl")