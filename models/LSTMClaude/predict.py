import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, "data"), "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Global variables for logging
DEBUG_LOG = False
debug_file = None
performance_file = None


def setup_logging():
    global debug_file, performance_file
    debug_file = open(os.path.join(RESULTS_DIR, "debug.log"), "a", encoding='utf-8')
    performance_file = open(os.path.join(RESULTS_DIR, "performance.log"), "a", encoding='utf-8')


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

    for i in range(len(data) - window_size + 1):
        seq = data[i:i + window_size]
        sequences.append(seq)

    return np.array(sequences)


def load_model_and_scalers():
    # Load model
    model_path = os.path.join(RESULTS_DIR, "best_model.pth")
    checkpoint = torch.load(model_path, map_location='cpu')
    model_params = checkpoint['model_params']

    model = LSTMModel(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load scalers
    scaler_path = os.path.join(RESULTS_DIR, "scalers.pkl")
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)

    log_debug("Model and scalers loaded successfully")

    return model, scalers['scaler_features'], scalers['scaler_close'], scalers['close_idx']


def load_and_preprocess_predict_data(file_path, window_size, scaler_features):
    log_debug(f"Loading prediction data from: {file_path}")

    df = pd.read_csv(file_path)
    df = df.sort_values('trade_date').reset_index(drop=True)

    # Store date and close price for output
    dates = df['trade_date'].values

    # Remove ts_code and use only numerical features
    feature_cols = [col for col in df.columns if col not in ['ts_code', 'trade_date']]
    data = df[feature_cols].values.astype(np.float32)

    log_debug(f"Prediction data shape: {data.shape}")
    log_debug(f"Feature columns: {feature_cols}")

    # Get close prices for evaluation
    close_idx = feature_cols.index('close')
    close_prices = data[:, close_idx].copy()

    # Normalize features using the same scaler from training
    data_scaled = scaler_features.transform(data)

    # Create sequences
    if len(data_scaled) >= window_size:
        X = create_sequences(data_scaled, window_size)
        # Corresponding dates and close prices for sequences
        seq_dates = dates[window_size - 1:]
        seq_close_prices = close_prices[window_size - 1:]
    else:
        log_debug("Warning: Not enough data for creating sequences")
        X = np.array([])
        seq_dates = np.array([])
        seq_close_prices = np.array([])

    log_debug(f"Prediction sequences shape: {X.shape}")

    return X, seq_dates, seq_close_prices, close_idx


def predict_stock_prices(model, X, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i:i + 1]).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            predictions.append(pred)

    return np.array(predictions)


def evaluate_predictions(predictions_real, true_real):
    mse = np.mean((predictions_real - true_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_real - true_real))
    mape = np.mean(np.abs((true_real - predictions_real) / true_real)) * 100

    log_performance("Prediction Evaluation Metrics:")
    log_performance(f"  MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    return rmse, mae, mape


def main():
    global DEBUG_LOG

    start_time = time.time()
    setup_logging()

    # Load configuration
    config_path = os.path.join(BASE_DIR, "model_args.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    DEBUG_LOG = config.get('debugLog', False)
    window_size = config['window_size']
    predict_file = os.path.join(DATA_DIR, config['predict'])

    log_performance("=== LSTM Stock Price Prediction Inference ===")
    log_performance(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    log_performance(f"Window size: {window_size}")
    log_performance(f"Prediction file: {config['predict']}")

    # Load model and scalers
    log_debug("Loading trained model and scalers...")
    model, scaler_features, scaler_close, close_idx = load_model_and_scalers()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load and preprocess prediction data
    X, dates, true_close_prices, _ = load_and_preprocess_predict_data(
        predict_file, window_size, scaler_features
    )

    if len(X) == 0:
        log_performance("Error: No valid sequences for prediction")
        return

    log_performance(f"Number of predictions to make: {len(X)}")

    # Make predictions
    log_debug("Making predictions...")
    predictions_scaled = predict_stock_prices(model, X, device)

    # Denormalize predictions
    predictions_real = scaler_close.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # Evaluate predictions
    if len(true_close_prices) > 0:
        evaluate_predictions(predictions_real, true_close_prices)

    # Save results to CSV
    results_df = pd.DataFrame({
        'trade_date': dates,
        'true_close': true_close_prices,
        'predicted_close': predictions_real
    })

    results_path = os.path.join(RESULTS_DIR, "predictions.csv")
    results_df.to_csv(results_path, index=False)

    log_performance(f"Predictions saved to: {results_path}")

    # Display sample results
    log_performance("\nSample Predictions:")
    log_performance("Date\t\tTrue\t\tPredicted\tDifference\tError%")
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        diff = row['predicted_close'] - row['true_close']
        error_pct = abs(diff / row['true_close']) * 100
        log_performance(
            f"{row['trade_date']}\t{row['true_close']:.2f}\t\t{row['predicted_close']:.2f}\t\t{diff:+.2f}\t\t{error_pct:.2f}%")

    end_time = time.time()
    prediction_time = end_time - start_time
    log_performance(f"\nTotal prediction time: {prediction_time:.2f} seconds")
    log_performance("Prediction completed successfully!")

    # Close log files
    if debug_file:
        debug_file.close()
    if performance_file:
        performance_file.close()


if __name__ == "__main__":
    main()