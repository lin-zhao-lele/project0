import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LSTM模型定义 (与train_model.py中相同)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")

    # 删除包含非法值的列
    invalid_columns = []
    for col in df.columns:
        if df[col].isnull().all() or df[col].dtype == 'object':
            invalid_columns.append(col)

    if invalid_columns:
        print(f"删除包含非法值的列: {invalid_columns}")
        df = df.drop(columns=invalid_columns)

    # 保存删除列的信息
    with open(os.path.join(RESULTS_DIR, "deleted_columns_predict.txt"), "w") as f:
        f.write(f"删除的列: {invalid_columns}\n")

    # 排除ts_code和trade_date，它们不是特征
    feature_columns = [col for col in df.columns if col not in ['ts_code', 'trade_date']]

    # 目标变量是close
    target_column = 'close'

    # 提取特征和目标变量
    features = df[feature_columns].drop(columns=[target_column])
    target = df[target_column]

    print(f"特征数量: {len(features.columns)}")
    print(f"特征列: {list(features.columns)}")

    return features, target, df['trade_date']


def create_sliding_window_dataset(features, target, window_size):
    """创建滑动窗口数据集"""
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i].values)
        y.append(target.iloc[i])
    return np.array(X), np.array(y)


def evaluate_model(model, X_test, y_test, scaler):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)

    # 反归一化
    test_outputs = scaler.inverse_transform(test_outputs.cpu().numpy())
    y_test_actual = scaler.inverse_transform(y_test.cpu().numpy())

    # 计算评估指标
    mse = mean_squared_error(y_test_actual, test_outputs)
    mae = mean_absolute_error(y_test_actual, test_outputs)
    r2 = r2_score(y_test_actual, test_outputs)

    # 对于分类指标，我们需要将问题转换为分类问题
    # 这里我们使用价格变化方向作为分类目标
    y_test_direction = np.diff(y_test_actual, axis=0)
    test_outputs_direction = np.diff(test_outputs, axis=0)

    # 将连续值转换为分类标签 (价格上涨=1, 价格下跌=0)
    y_test_binary = (y_test_direction > 0).astype(int)
    test_outputs_binary = (test_outputs_direction > 0).astype(int)

    # 确保两个数组长度相同
    min_len = min(len(y_test_binary), len(test_outputs_binary))
    y_test_binary = y_test_binary[:min_len]
    test_outputs_binary = test_outputs_binary[:min_len]

    if len(y_test_binary) > 0 and len(np.unique(y_test_binary)) > 1:
        accuracy = accuracy_score(y_test_binary, test_outputs_binary)
        precision = precision_score(y_test_binary, test_outputs_binary, zero_division=0)
        recall = recall_score(y_test_binary, test_outputs_binary, zero_division=0)
        f1 = f1_score(y_test_binary, test_outputs_binary, zero_division=0)

        # 计算AUC
        try:
            auc = roc_auc_score(y_test_binary, test_outputs_binary)
        except ValueError:
            auc = float('nan')
    else:
        accuracy = precision = recall = f1 = auc = float('nan')

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    }

    return metrics, test_outputs, y_test_actual


def main():
    # 加载配置
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)

    # 加载并预处理数据
    predict_file = os.path.join(DATA_DIR, config['predict'])
    features, target, trade_dates = load_and_preprocess_data(predict_file)

    # 加载训练时的归一化器
    # 注意：在实际应用中，应该保存训练时的归一化器
    # 这里我们重新拟合归一化器以演示流程
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    # 转换为DataFrame以便处理
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    target_scaled_series = pd.Series(target_scaled, name=target.name)

    # 获取模型参数
    params = config['params']
    window_size = params['window_size']

    # 创建滑动窗口数据集
    X_predict, y_predict = create_sliding_window_dataset(features_scaled_df, target_scaled_series, window_size)

    # 转换为PyTorch张量
    X_predict_tensor = torch.FloatTensor(X_predict).to(device)
    y_predict_tensor = torch.FloatTensor(y_predict).reshape(-1, 1).to(device)

    # 初始化模型
    input_size = X_predict_tensor.shape[2]
    model = LSTMModel(input_size, params['hidden_size'], params['num_layers'], 1, params['dropout']).to(device)

    # 加载训练好的模型
    model_path = os.path.join(RESULTS_DIR, 'lstm_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        print(f"模型文件 {model_path} 不存在，请先运行训练脚本")
        return

    # 评估模型
    metrics, predictions, actuals = evaluate_model(model, X_predict_tensor, y_predict_tensor, target_scaler)

    # 保存评估指标
    metrics_path = os.path.join(RESULTS_DIR, 'predict_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("预测集评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\n评估指标已保存到: {metrics_path}")

    # 保存预测结果
    # 确保trade_dates与预测结果对齐
    # 由于滑动窗口，我们需要跳过前window_size个数据点
    trade_dates_aligned = trade_dates[window_size:window_size + len(predictions)]

    results_df = pd.DataFrame({
        'trade_date': trade_dates_aligned,
        'true_close': actuals.flatten(),
        'predicted_close': predictions.flatten()
    })

    results_path = os.path.join(RESULTS_DIR, 'predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n预测结果已保存到: {results_path}")


if __name__ == "__main__":
    main()