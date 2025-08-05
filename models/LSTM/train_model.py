import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
import optuna
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LSTM模型定义
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
    with open(os.path.join(RESULTS_DIR, "deleted_columns.txt"), "w") as f:
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


def train_model_with_optuna(X_train, y_train, X_val, y_val, config):
    """使用Optuna进行超参数优化"""

    def objective(trial):
        # 建议超参数
        window_size = trial.suggest_int('window_size', 5, 60)
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        num_epochs = trial.suggest_int('num_epochs', 10, 100)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        # 创建滑动窗口数据集
        X_train_window, y_train_window = create_sliding_window_dataset(X_train, y_train, window_size)
        X_val_window, y_val_window = create_sliding_window_dataset(X_val, y_val, window_size)

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train_window).to(device)
        y_train_tensor = torch.FloatTensor(y_train_window).reshape(-1, 1).to(device)
        X_val_tensor = torch.FloatTensor(X_val_window).to(device)
        y_val_tensor = torch.FloatTensor(y_val_window).reshape(-1, 1).to(device)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化模型
        input_size = X_train_tensor.shape[2]
        model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        model.train()
        for epoch in range(num_epochs):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        return val_loss.item()

    # 创建Optuna研究
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # 减少试验次数以加快速度

    # 保存最佳参数
    best_params = study.best_params
    print(f"最佳参数: {best_params}")

    return best_params


def train_model_with_fixed_params(X_train, y_train, X_val, y_val, params):
    """使用固定参数训练模型"""
    window_size = params['window_size']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    dropout = params['dropout']

    # 创建滑动窗口数据集
    X_train_window, y_train_window = create_sliding_window_dataset(X_train, y_train, window_size)
    X_val_window, y_val_window = create_sliding_window_dataset(X_val, y_val, window_size)

    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_window).to(device)
    y_train_tensor = torch.FloatTensor(y_train_window).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_window).to(device)
    y_val_tensor = torch.FloatTensor(y_val_window).reshape(-1, 1).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    input_size = X_train_tensor.shape[2]
    model = LSTMModel(input_size, hidden_size, num_layers, 1, dropout).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model


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
    train_file = os.path.join(DATA_DIR, config['training'])
    features, target, trade_dates = load_and_preprocess_data(train_file)

    # 数据归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    # 转换为DataFrame以便处理
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
    target_scaled_series = pd.Series(target_scaled, name=target.name)

    # 分割训练集和验证集 (80% 训练, 20% 验证)
    split_idx = int(len(features_scaled_df) * 0.8)
    X_train = features_scaled_df[:split_idx]
    y_train = target_scaled_series[:split_idx]
    X_val = features_scaled_df[split_idx:]
    y_val = target_scaled_series[split_idx:]

    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")

    # 根据配置决定是否进行超参数优化
    if config['auto_tune']:
        print("开始超参数优化...")
        best_params = train_model_with_optuna(X_train, y_train, X_val, y_val, config)

        # 将最佳参数写入配置文件
        config['params'] = best_params
        with open(os.path.join(BASE_DIR, 'model_args.json'), 'w') as f:
            json.dump(config, f, indent=4)
    else:
        print("使用配置文件中的参数训练模型...")
        best_params = config['params']

    # 使用最佳参数训练最终模型
    print("使用最佳参数训练最终模型...")
    model = train_model_with_fixed_params(X_train, y_train, X_val, y_val, best_params)

    # 保存模型
    model_path = os.path.join(RESULTS_DIR, 'lstm_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # 评估模型
    X_val_window, y_val_window = create_sliding_window_dataset(X_val, y_val, best_params['window_size'])
    X_val_tensor = torch.FloatTensor(X_val_window).to(device)
    y_val_tensor = torch.FloatTensor(y_val_window).reshape(-1, 1).to(device)

    metrics, predictions, actuals = evaluate_model(model, X_val_tensor, y_val_tensor, target_scaler)

    # 保存评估指标
    metrics_path = os.path.join(RESULTS_DIR, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("训练集评估指标:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\n评估指标已保存到: {metrics_path}")


if __name__ == "__main__":
    main()