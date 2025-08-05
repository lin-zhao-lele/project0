import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "results")


def load_model():
    """
    加载训练好的模型
    """
    model_path = os.path.join(MODEL_DIR, 'credit_risk_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = joblib.load(model_path)
    return model


def load_inference_data():
    """
    加载推理数据
    """
    data_path = os.path.join(DATA_DIR, 'inference_data.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"推理数据文件不存在: {data_path}")
    
    data = pd.read_csv(data_path)
    return data


def save_predictions(predictions, probabilities, data):
    """
    保存预测结果
    """
    # 创建结果DataFrame
    results = data.copy()
    results['risk_prediction'] = predictions
    results['risk_probability'] = probabilities
    
    # 保存结果
    output_path = os.path.join(DATA_DIR, 'predictions.csv')
    results.to_csv(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")


def main():
    """
    主函数
    """
    # 加载模型
    print("正在加载模型...")
    model = load_model()
    
    # 加载推理数据
    print("正在加载推理数据...")
    data = load_inference_data()
    
    # 准备特征数据
    # 注意：确保特征列与训练时一致
    feature_columns = [col for col in data.columns if col not in ['risk_flag']]
    X = data[feature_columns]
    
    # 进行预测
    print("正在进行预测...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # 获取正类概率
    
    # 保存预测结果
    save_predictions(predictions, probabilities, data)
    
    # 打印一些预测统计信息
    risk_count = np.sum(predictions)
    total_count = len(predictions)
    risk_rate = risk_count / total_count
    
    print(f"\n预测统计信息:")
    print(f"总样本数: {total_count}")
    print(f"风险用户数: {risk_count}")
    print(f"风险用户比例: {risk_rate:.2%}")
    
    # 如果推理数据包含真实标签，计算并显示模型性能指标
    if 'risk_flag' in data.columns:
        y_true = data['risk_flag']
        mse = mean_squared_error(y_true, predictions)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        
        print(f"\n模型性能指标:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    main()