import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 工程根目录 = 脚本所在目录的上上级
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)


def load_model():
    """
    加载训练好的模型和标准化器
    """
    model = joblib.load(os.path.join(DATA_DIR, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
    return model, scaler


def load_inference_data():
    """
    加载推理数据
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'inference_data.csv'))
    return data


def load_test_data():
    """
    加载测试数据
    """
    data = pd.read_csv(os.path.join(DATA_DIR, 'test_data.csv'))
    X = data.drop('risk_flag', axis=1)
    y = data['risk_flag']
    return X, y


def make_predictions():
    """
    使用训练好的模型进行预测
    """
    # 加载模型和标准化器
    model, scaler = load_model()

    # 加载测试数据
    X_test, y_test = load_test_data()

    # 标准化特征
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # 进行预测
    y_pred = model.predict(X_test_scaled)

    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("测试集性能指标:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # 保存预测结果
    results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    results.to_csv(os.path.join(DATA_DIR, 'predictions.csv'), index=False)
    print(f"\n预测结果已保存到 predictions.csv，共 {len(results)} 条记录")

    # 加载推理数据并进行预测
    inference_data = load_inference_data()
    inference_data_scaled = scaler.transform(inference_data)
    inference_data_scaled = pd.DataFrame(inference_data_scaled, columns=inference_data.columns)
    inference_pred = model.predict(inference_data_scaled)

    # 保存推理结果
    inference_results = inference_data.copy()
    inference_results['risk_prediction'] = inference_pred
    inference_results.to_csv(os.path.join(DATA_DIR, 'inference_predictions.csv'), index=False)
    print(f"推理预测结果已保存到 inference_predictions.csv，共 {len(inference_results)} 条记录")


if __name__ == "__main__":
    make_predictions()