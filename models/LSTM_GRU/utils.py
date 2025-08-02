# utils.py (PyTorch version with device support)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import optuna

# 滑动窗口构造序列数据
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size][3])  # 第4列是close
    return np.array(X), np.array(y)

# PyTorch LSTM 模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 构建模型
def build_model(input_shape, hidden_units):
    input_size = input_shape[1]
    model = LSTMModel(input_size, hidden_units)
    return model

# 自动调参（Optuna）支持传入 device
def tune_lstm_model(X_train, y_train, X_val, y_val, device):
    def objective(trial):
        hidden_units = trial.suggest_int("hidden_units", 32, 128, step=16)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        epochs = 20

        model = LSTMModel(X_train.shape[2], hidden_units).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for xb, yb in dataloader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        val_mse = mean_squared_error(y_val, val_preds)
        return val_mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # 用最优参数重建模型
    best_model = LSTMModel(X_train.shape[2], best_params["hidden_units"]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    )
    dataloader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

    best_model.train()
    for epoch in range(20):
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = best_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return best_model, best_params
