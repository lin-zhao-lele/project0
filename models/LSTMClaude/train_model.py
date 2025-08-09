import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import optuna
import pickle
from datetime import datetime

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global variables for logging
DEBUG_LOG = False
debug_file = None
performance_file = None


def setup_logging():
    global debug_file, performance_file
    debug_file = open(os.path.join(RESULTS_DIR, "debug.log"), "w", encoding='utf-8')
    performance_file = open(os.path.join(RESULTS_DIR, "performance.log"), "w", encoding='utf-8')


def log_debug(message):
    if DEBUG_LOG and debug_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_file.write(f"[{timestamp}] {message}\n")
        debug_file.flush()


def log_performance(message):
    if performance_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        performance_file.write(f"[{timestamp}] {message}\n")
        performance_file.flush()
        print(message)


class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out[:, -1, :])
        output = self.fc(lstm_out)
        return output


def create_sequences(data, window_size):
    sequences = []
    targets = []

    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        target = data[i + window_size]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def load_and_preprocess_data(file_path, window_size):
    log_debug(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)
    df = df.sort_values('trade_date').reset_index(drop=True)

    # Remove ts_code and use only numerical features
    feature_cols = [col for col in df.columns if col not in ['ts_code', 'trade_date']]
    data = df[feature_cols].values.astype(np.float32)

    log_debug(f"Data shape: {data.shape}")
    log_debug(f"Feature columns: {feature_cols}")

    # Get close price index
    close_idx = feature_cols.index('close')
    close_prices = data[:, close_idx].copy()

    # Normalize features
    scaler_features = StandardScaler()
    data_scaled = scaler_features.fit_transform(data)

    # Normalize close prices separately for inverse transform
    scaler_close = StandardScaler()
    close_scaled = scaler_close.fit_transform(close_prices.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = create_sequences(data_scaled, window_size)

    log_debug(f"Sequences shape: {X.shape}, Targets shape: {y.shape}")

    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Get corresponding close prices for evaluation
    close_prices_seq = close_prices[window_size:]
    close_train = close_prices_seq[:train_size]
    close_val = close_prices_seq[train_size:]

    log_debug(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    return (X_train, y_train, X_val, y_val,
            close_train, close_val,
            scaler_features, scaler_close,
            feature_cols.index('close'))


def train_model(model, train_loader, val_loader, device, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch % 20 == 0:
            log_debug(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Load best model state
    model.load_state_dict(best_model_state)
    return best_val_loss


def evaluate_model(model, X, y, close_true, scaler_close, close_idx, device, phase=""):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i:i + 1]).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            predictions.append(pred)

    predictions = np.array(predictions)

    # Calculate metrics on normalized values
    mse = np.mean((predictions - y) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y))

    # Denormalize for real price metrics
    pred_real = scaler_close.inverse_transform(predictions.reshape(-1, 1)).flatten()
    true_real = close_true

    mse_real = np.mean((pred_real - true_real) ** 2)
    rmse_real = np.sqrt(mse_real)
    mae_real = np.mean(np.abs(pred_real - true_real))
    mape_real = np.mean(np.abs((true_real - pred_real) / true_real)) * 100

    log_performance(f"{phase} Evaluation Metrics:")
    log_performance(f"  Normalized - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    log_performance(
        f"  Real Price - MSE: {mse_real:.2f}, RMSE: {rmse_real:.2f}, MAE: {mae_real:.2f}, MAPE: {mape_real:.2f}%")

    return rmse_real


def objective(trial):
    # Hyperparameter suggestions
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    try:
        # Load data
        config_path = os.path.join(BASE_DIR, "model_args.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        window_size = config['window_size']
        training_file = os.path.join(DATA_DIR, config['training'])

        (X_train, y_train, X_val, y_val,
         close_train, close_val,
         scaler_features, scaler_close, close_idx) = load_and_preprocess_data(training_file, window_size)

        # Create datasets and dataloaders
        train_dataset = StockDataset(X_train, y_train[:, close_idx])
        val_dataset = StockDataset(X_val, y_val[:, close_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X_train.shape[2]
        model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(device)

        # Train model
        val_loss = train_model(model, train_loader, val_loader, device, 100, learning_rate)

        log_debug(f"Trial {trial.number}: Val Loss = {val_loss:.6f}, Params = {trial.params}")

        return val_loss

    except Exception as e:
        log_debug(f"Trial {trial.number} failed: {str(e)}")
        return float('inf')


def main():
    global DEBUG_LOG

    start_time = time.time()
    setup_logging()

    # Load configuration
    config_path = os.path.join(BASE_DIR, "model_args.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    DEBUG_LOG = config.get('debugLog', False)

    log_performance("=== LSTM Stock Price Prediction Training ===")
    log_performance(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log_debug("Starting hyperparameter optimization...")

    # Hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    log_performance(f"Best hyperparameters: {best_params}")

    # Train final model with best parameters
    window_size = config['window_size']
    training_file = os.path.join(DATA_DIR, config['training'])

    (X_train, y_train, X_val, y_val,
     close_train, close_val,
     scaler_features, scaler_close, close_idx) = load_and_preprocess_data(training_file, window_size)

    # Create datasets
    train_dataset = StockDataset(X_train, y_train[:, close_idx])
    val_dataset = StockDataset(X_val, y_val[:, close_idx])

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)

    # Create and train final model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[2]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(device)

    log_performance("Training final model...")
    train_model(model, train_loader, val_loader, device, 200, best_params['learning_rate'])

    # Evaluate model
    log_performance("\nEvaluating model performance...")
    evaluate_model(model, X_train, y_train[:, close_idx], close_train, scaler_close, close_idx, device, "Training")
    evaluate_model(model, X_val, y_val[:, close_idx], close_val, scaler_close, close_idx, device, "Validation")

    # Save model and scalers
    model_path = os.path.join(RESULTS_DIR, "best_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': {
            'input_size': input_size,
            'hidden_size': best_params['hidden_size'],
            'num_layers': best_params['num_layers'],
            'dropout': best_params['dropout']
        }
    }, model_path)

    # Save scalers
    scaler_path = os.path.join(RESULTS_DIR, "scalers.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({
            'scaler_features': scaler_features,
            'scaler_close': scaler_close,
            'close_idx': close_idx
        }, f)

    # Update config with best parameters
    config['best_params'] = best_params
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    end_time = time.time()
    training_time = end_time - start_time
    log_performance(f"\nTotal training time: {training_time:.2f} seconds")
    log_performance("Training completed successfully!")

    # Close log files
    if debug_file:
        debug_file.close()
    if performance_file:
        performance_file.close()


if __name__ == "__main__":
    main()