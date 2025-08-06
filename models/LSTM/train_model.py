"""
LSTM股票价格预测模型 - 训练脚本

本脚本实现了一个基于LSTM的股票价格预测模型，主要功能包括：
1. 数据加载和预处理
2. LSTM模型定义
3. 滑动窗口数据生成
4. 使用Optuna进行超参数优化
5. 模型训练和验证

适合初学者学习LSTM模型的实现原理和PyTorch的基本用法
"""
import os
import time
import torch
import torch.nn as nn  # PyTorch神经网络模块
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
import optuna  # 超参数优化库
import json  # JSON文件处理

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
LSTM模型定义

这是一个标准的LSTM网络结构，包含以下组件：
1. LSTM层：处理时序数据，提取时间序列特征
2. 全连接层：将LSTM输出映射到预测值

参数说明：
- input_size: 输入特征维度
- hidden_size: LSTM隐藏层维度
- num_layers: LSTM堆叠层数
- output_size: 输出维度(这里预测收盘价，所以是1)
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  # LSTM隐藏层大小
        self.num_layers = num_layers    # LSTM层数
        # 定义LSTM层，batch_first=True表示输入数据的第一个维度是batch_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，将LSTM输出映射到预测值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播过程
        
        参数:
        x: 输入数据，形状为(batch_size, seq_len, input_size)
        
        返回:
        预测值，形状为(batch_size, output_size)
        """
        # 初始化隐藏状态和细胞状态(全零初始化)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out形状: (batch_size, seq_len, hidden_size)
        
        # 只取最后一个时间步的输出，通过全连接层得到预测值
        out = self.fc(out[:, -1, :])
        return out

"""
数据加载和预处理函数

从CSV文件加载股票数据，并进行以下处理：
1. 排除不需要的特征(股票代码和交易日期)
2. 将收盘价作为预测目标
3. 对特征和目标值进行归一化(缩放到0-1之间)

参数:
filename: 数据文件名

返回:
features: 归一化后的特征数据
target: 归一化后的目标值(收盘价)
feature_scaler: 特征归一化器(用于后续数据转换)
target_scaler: 目标值归一化器
"""
def load_data(filename):
    # 读取CSV文件
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    
    # 特征处理：排除股票代码和交易日期，收盘价作为目标
    features = df.drop(columns=['ts_code', 'trade_date', 'close'])  # 特征数据
    target = df['close'].values.reshape(-1, 1)  # 目标值(收盘价)
    
    # 数据归一化：将特征和目标值缩放到0-1范围
    feature_scaler = MinMaxScaler()  # 特征归一化器
    target_scaler = MinMaxScaler()   # 目标值归一化器
    
    # 拟合并转换数据
    features = feature_scaler.fit_transform(features)
    target = target_scaler.fit_transform(target)
    
    return features, target, feature_scaler, target_scaler

"""
创建滑动窗口数据集

将时序数据转换为适合LSTM训练的滑动窗口格式
例如：用前30天的数据预测第31天的收盘价

参数:
features: 特征数据
target: 目标值
window_size: 滑动窗口大小(默认30天)

返回:
X: 滑动窗口特征数据，形状为(n_samples, window_size, n_features)
y: 对应的目标值
"""
def create_dataset(features, target, window_size=30):
    X, y = [], []
    # 从window_size开始，避免越界
    for i in range(window_size, len(features)):
        # 取前window_size天的特征作为输入
        X.append(features[i-window_size:i])
        # 第i天的收盘价作为目标
        y.append(target[i])
    return np.array(X), np.array(y)

"""
模型训练函数(供Optuna调用的目标函数)

Optuna会多次调用此函数，尝试不同的超参数组合
最终选择验证集上表现最好的参数

参数:
trial: Optuna的Trial对象，用于建议超参数

返回:
验证集上的损失值(Optuna会最小化此值)
"""
def train_model(trial):
    # Optuna建议的超参数范围
    hidden_size = trial.suggest_int('hidden_size', 32, 256)  # LSTM隐藏层大小
    num_layers = trial.suggest_int('num_layers', 1, 3)       # LSTM层数
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)  # 学习率
    batch_size = trial.suggest_int('batch_size', 16, 128)    # 批大小
    num_epochs = trial.suggest_int('num_epochs', 10, 100)   # 训练轮数
    
    # 加载配置和数据
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    
    # 加载并预处理训练数据
    features, target, feature_scaler, target_scaler = load_data(config['training'])
    
    # 划分训练集和验证集(80%训练，20%验证)
    split_idx = int(0.8 * len(features))
    train_X, train_y = create_dataset(features[:split_idx], target[:split_idx])
    val_X, val_y = create_dataset(features[split_idx:], target[split_idx:])
    
    # 转换为PyTorch张量并移动到相应设备(CPU/GPU)
    train_X = torch.FloatTensor(train_X).to(device)
    train_y = torch.FloatTensor(train_y).to(device)
    val_X = torch.FloatTensor(val_X).to(device)
    val_y = torch.FloatTensor(val_y).to(device)
    
    # 创建模型
    input_size = train_X.shape[2]  # 输入特征维度
    model = LSTMModel(input_size, hidden_size, num_layers, 1).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        outputs = model(train_X)  # 前向传播
        loss = criterion(outputs, train_y)  # 计算损失
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        # 验证
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y)
    
    return val_loss.item()  # 返回验证集上的损失值

if __name__ == '__main__':
    start_time = time.time()
    
    # Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=50)
    
    # Save best model
    best_params = study.best_params
    
    # Log performance
    with open(os.path.join(RESULTS_DIR, 'performance.log'), 'a') as f:
        f.write(f"Training time: {time.time() - start_time:.2f} seconds\n")
        f.write(f"Best validation loss: {study.best_value:.6f}\n")
        f.write(f"Best parameters: {best_params}\n")
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Best validation loss: {study.best_value:.6f}")
    print(f"Best parameters: {best_params}")
    
    # 将最佳参数和窗口大小保存到model_args.json
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    config['best_params'] = best_params
    config['window_size'] = 30  # 统一窗口大小
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 使用最佳参数重新训练并保存模型
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    
    # 加载并预处理训练数据
    features, target, feature_scaler, target_scaler = load_data(config['training'])
    
    # 创建完整训练集
    X, y = create_dataset(features, target)
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    
    # 创建并训练最佳模型
    input_size = X.shape[2]
    best_model = LSTMModel(input_size, best_params['hidden_size'], 
                          best_params['num_layers'], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
    
    # 训练循环
    for epoch in range(best_params['num_epochs']):
        best_model.train()
        outputs = best_model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 保存模型
    torch.save(best_model.state_dict(), os.path.join(BASE_DIR, 'model.pth'))
    print("Model saved to model.pth")