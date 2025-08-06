"""
LSTM股票价格预测模型 - 预测脚本

本脚本实现以下功能：
1. 加载训练好的LSTM模型
2. 加载并预处理预测数据
3. 使用滑动窗口技术进行预测
4. 反归一化预测结果
5. 保存预测结果和性能指标

适合初学者学习LSTM模型的预测流程
"""
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from module.visualization.pltTrend import plot_trend_signals_from_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
LSTM模型定义(与train_model.py中相同)

这是一个标准的LSTM网络结构，包含以下组件：
1. LSTM层：处理时序数据
2. 全连接层：将LSTM输出映射到预测值

参数说明：
- input_size: 输入特征维度
- hidden_size: LSTM隐藏层维度
- num_layers: LSTM堆叠层数
- output_size: 输出维度(这里预测收盘价，所以是1)
"""
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size  # LSTM隐藏层大小
        self.num_layers = num_layers    # LSTM层数
        # 定义LSTM层，batch_first=True表示输入数据的第一个维度是batch_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层，将LSTM输出映射到预测值
        self.fc = torch.nn.Linear(hidden_size, output_size)

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
数据加载和归一化函数

加载预测数据并使用训练数据的归一化参数进行归一化
确保预测数据的归一化方式与训练数据一致

参数:
filename: 预测数据文件名

返回:
df: 原始数据DataFrame
features: 归一化后的特征数据
target: 归一化后的目标值
feature_scaler: 特征归一化器
target_scaler: 目标值归一化器
"""
def load_data_and_scalers(filename):
    # 读取预测数据
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    
    # 加载训练时使用的归一化器
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # 使用训练数据拟合归一化器(确保归一化方式一致)
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    
    # 加载训练数据
    train_df = pd.read_csv(os.path.join(DATA_DIR, config['training']))
    train_features = train_df.drop(columns=['ts_code', 'trade_date', 'close'])
    train_target = train_df['close'].values.reshape(-1, 1)
    
    # 拟合归一化器
    feature_scaler.fit(train_features)
    target_scaler.fit(train_target)
    
    # 对预测数据进行相同的归一化处理(确保列顺序与训练时一致)
    feature_columns = ['open', 'high', 'low', 'vol', 'amount']  # 明确指定特征列顺序
    features = df[feature_columns]
    target = df['close'].values.reshape(-1, 1)
    
    features = feature_scaler.transform(features)
    target = target_scaler.transform(target)
    
    return df, features, target, feature_scaler, target_scaler

"""
创建滑动窗口数据集

将时序数据转换为适合LSTM预测的滑动窗口格式
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

if __name__ == '__main__':
    """
    主预测流程
    
    1. 加载配置文件
    2. 加载并预处理数据
    3. 创建滑动窗口数据集
    4. 加载模型(实际项目中应加载训练好的模型)
    5. 进行预测
    6. 反归一化预测结果
    7. 保存结果和性能指标
    """
    # 加载配置文件
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    
    # 加载并预处理预测数据
    df, features, target, feature_scaler, target_scaler = load_data_and_scalers(config['predict'])
    
    # 创建滑动窗口数据集
    X, y = create_dataset(features, target)
    
    # 加载训练好的模型
    input_size = X.shape[2]  # 输入特征维度
    
    # 从model_args.json加载训练时使用的模型参数和窗口大小
    with open(os.path.join(BASE_DIR, 'model_args.json'), 'r') as f:
        config = json.load(f)
    window_size = config['window_size']  # 从配置文件读取窗口大小
    
    # 使用与训练时相同的模型结构
    model = LSTMModel(input_size, 
                     config['best_params']['hidden_size'], 
                     config['best_params']['num_layers'], 
                     1).to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'model.pth')))  # 加载模型权重
    
    # 进行预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # 反归一化预测结果(还原为真实股价)
    predictions = target_scaler.inverse_transform(predictions)
    true_values = target_scaler.inverse_transform(y)
    
    # 准备结果
    # 预测结果对应的是输入窗口后的第一个时间点
    # 所以预测结果应该与原始数据的[window_size:]对应
    results = df.copy()
    results['true_close'] = np.nan  # 先初始化为NaN
    results['predicted_close'] = np.nan
    
    # 将预测结果和真实值填充到正确位置
    # 预测结果对应的是第window_size+1天到最后一天
    # 注意：true_values和predictions的长度应为len(df)-window_size
    
    # 检查长度是否匹配
    if len(predictions) != len(df) - window_size:
        raise ValueError(f"预测结果长度不匹配: predictions长度={len(predictions)}, 预期长度={len(df)-window_size}")
    
    # 填充真实值和预测值
    start_idx = window_size
    end_idx = len(df)  # 填充到最后一天
    results.iloc[start_idx:end_idx, results.columns.get_loc('true_close')] = true_values.flatten()
    results.iloc[start_idx:end_idx, results.columns.get_loc('predicted_close')] = predictions.flatten()
    
    # 确保前window_size天没有预测值
    results.iloc[:window_size, results.columns.get_loc('predicted_close')] = np.nan
    
    # 按日期排序并保存预测结果
    results = results.sort_values('trade_date')
    results[['trade_date', 'true_close', 'predicted_close']].to_csv(
        os.path.join(RESULTS_DIR, 'predictions.csv'), index=False)
    
    # 计算并记录均方误差(MSE)
    mse = np.mean((results['true_close'] - results['predicted_close'])**2)
    with open(os.path.join(RESULTS_DIR, 'performance.log'), 'a') as f:
        f.write(f"Prediction MSE: {mse:.6f}\n")
    
    # 打印结果信息
    print(f"Predictions saved to {os.path.join(RESULTS_DIR, 'predictions.csv')}")
    print(f"Prediction MSE: {mse:.6f}")

    #################################################################
    # ========== 绘图 ==========
    df_all_out = pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv"))

    df_all_out["trade_date"] = pd.to_datetime(df_all_out["trade_date"], format="%Y%m%d")
    plt.figure(figsize=(10, 4))

    if "true_close" in df_all_out.columns:
        plt.plot(df_all_out["trade_date"], df_all_out["true_close"], label="True Close")
    plt.plot(df_all_out["trade_date"], df_all_out["predicted_close"], label="Predicted Close")

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("LSTM Inference Prediction with Future Forecast")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "lstm_inference_plot.png")
    plt.savefig(plot_path)
    print(f"📊 推理图保存至 {plot_path}")


    #################################################################
    # 阈值：预测涨跌幅小于该值时忽略信号（单位：百分比）
    threshold_pct = 0.005  # 0.5%

    # 读取预测文件
    pred_df = pd.read_csv(os.path.join(RESULTS_DIR, "predictions.csv"))

    # 计算预测涨跌幅
    pred_df["predicted_change"] = pred_df["predicted_close"].diff() / pred_df["predicted_close"].shift(1)
    pred_df["true_change"] = pred_df["true_close"].diff() / pred_df["true_close"].shift(1)

    # 生成预测趋势信号（1 = 上涨，-1 = 下跌，0 = 无操作）
    pred_df["trend_signal"] = 0
    pred_df.loc[pred_df["predicted_change"] > threshold_pct, "trend_signal"] = 1
    pred_df.loc[pred_df["predicted_change"] < -threshold_pct, "trend_signal"] = -1

    # 生成真实趋势方向（用于验证）
    pred_df["true_trend"] = 0
    pred_df.loc[pred_df["true_change"] > 0, "true_trend"] = 1
    pred_df.loc[pred_df["true_change"] < 0, "true_trend"] = -1

    # 计算趋势方向准确率
    valid_mask = pred_df["trend_signal"] != 0
    accuracy = (pred_df.loc[valid_mask, "trend_signal"] == pred_df.loc[valid_mask, "true_trend"]).mean()

    print(f"趋势信号准确率（过滤小波动后）: {accuracy:.2%}")
    print(f"总信号数: {valid_mask.sum()} 条")

    # 保存信号文件
    trend_path = os.path.join(RESULTS_DIR, "lstm_inference_trend_signals.csv")
    pred_df.to_csv(trend_path, index=False)
    print(f"趋势信号已保存到 {trend_path}")

    plot_trend_signals_from_csv(trend_path, os.path.join(RESULTS_DIR, "lstm_inference_trend_signals_plot.png"))
