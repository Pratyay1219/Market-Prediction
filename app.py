import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility for academic evaluation
np.random.seed(42)
tf.random.set_seed(42)

# ---------------- APP CONFIG ----------------
st.set_page_config(page_title="Educational Price Prediction – ML & LSTM", layout="centered")

st.title("📈 Educational Price Prediction System")
st.caption("Academic demonstration of predictive analytics on historical data — NOT trading or financial advice.")

# ---------------- DATASET REGISTRY ----------------
DATASETS = {
    "BankNifty": "data/BANKNIFTY_FUT.csv",
    "Nifty": "data/NIFTY_FUT.csv",
    "Gold": "data/GOLD_FUT.csv",
    "Silver": "data/SILVER_FUT.csv",
    "Crude Oil": "data/CRUDEOIL_FUT.csv",
    "Natural Gas": "data/NATURALGAS_FUT.csv"
}

st.info(
    "This app predicts the **next trading day's end-of-day close** using historical prices. "
    "It is designed for learning concepts like preprocessing, regression, model evaluation, and sequence models."
)

# ---------------- HELPER FUNCTIONS ----------------
def load_data(path: str) -> pd.DataFrame:
    """Load and sort EOD data. Only datetime and close are used for prediction in this app."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    return df[["datetime", "close"]]


def scale_close_series(close_values: np.ndarray, train_ratio: float = 0.8):
    """
    Scale close prices with MinMaxScaler fit ONLY on the training portion
    to avoid look-ahead bias (time-series safe scaling).
    """
    scaler = MinMaxScaler()
    split_idx = int(len(close_values) * train_ratio)
    train_close = close_values[:split_idx]
    test_close = close_values[split_idx:]

    scaled_train = scaler.fit_transform(train_close)
    scaled_test = scaler.transform(test_close)
    scaled_all = np.vstack([scaled_train, scaled_test])
    return scaled_all, scaler


def create_sequences(data: np.ndarray, window: int):
    """
    Convert a 1D/2D time series into supervised sequences:
    X = past 'window' values, y = next value.
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """Simple LSTM model for experimental sequence prediction."""
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def build_classical_model(model_name: str):
    """Factory for classical regression models used as baselines."""
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Polynomial Regression (Degree 2)":
        return Pipeline(
            [
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lin_reg", LinearRegression()),
            ]
        )
    if model_name == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


# ---------------- USER INPUT ----------------
instrument = st.selectbox("Select Instrument", list(DATASETS.keys()))

window_size = st.slider(
    "Look-back Window (Days used as input features)",
    min_value=5,
    max_value=20,
    value=10,
)

tab_ml, tab_lstm = st.tabs(
    ["Classical ML (Baseline – Regression)", "LSTM (Experimental – Neural Network)"]
)

# ---------------- CLASSICAL ML TAB ----------------
with tab_ml:
    st.subheader("Classical Regression Models (Baseline)")
    st.write(
        "These models demonstrate **supervised learning, bias–variance trade-off, and model evaluation** "
        "using MAE, MSE, RMSE, and R²."
    )

    model_name = st.selectbox(
        "Choose Regression Model",
        ["Linear Regression", "Polynomial Regression (Degree 2)", "Random Forest Regressor"],
    )

    if st.button("Run Classical ML Baseline"):
        with st.spinner("Training classical regression model..."):
            df = load_data(DATASETS[instrument])
            close_values = df[["close"]].values  # shape (n_samples, 1)

            # Time-series safe scaling
            scaled_close, scaler = scale_close_series(close_values, train_ratio=0.8)

            # Create sequences and flatten for scikit-learn (samples, window)
            X, y = create_sequences(scaled_close, window_size)
            X = X.reshape(X.shape[0], X.shape[1])  # drop last dimension (1)

            # Time-based split (no shuffling)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = build_classical_model(model_name)
            model.fit(X_train, y_train.ravel())

            # Evaluation on test set
            preds_scaled = model.predict(X_test)
            preds_scaled_2d = preds_scaled.reshape(-1, 1)
            y_test_2d = y_test.reshape(-1, 1)

            preds = scaler.inverse_transform(preds_scaled_2d).ravel()
            y_true = scaler.inverse_transform(y_test_2d).ravel()

            mae = mean_absolute_error(y_true, preds)
            mse = mean_squared_error(y_true, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, preds)

            # Predict next trading day's close
            last_seq = scaled_close[-window_size:].reshape(1, window_size)
            next_scaled = model.predict(last_seq)
            next_price = scaler.inverse_transform(
                np.array(next_scaled).reshape(-1, 1)
            )[0, 0]

        st.success("Classical ML baseline prediction complete")

        st.metric(
            label="Predicted Next Day Close (Baseline Model)",
            value=round(float(next_price), 2),
        )

        st.write("**Instrument:**", instrument)
        st.write("**Model (Classical):**", model_name)
        st.write("**MAE:**", round(float(mae), 4))
        st.write("**MSE:**", round(float(mse), 4))
        st.write("**RMSE:**", round(float(rmse), 4))
        st.write("**R² Score:**", round(float(r2), 4))
        st.write("**Last Available Date:**", df.iloc[-1]["datetime"].date())

        st.caption(
            "These results are for **educational comparison of regression models** on historical data, "
            "not for live trading or financial decision-making."
        )

# ---------------- LSTM TAB (EXPERIMENTAL) ----------------
with tab_lstm:
    st.subheader("LSTM Neural Network (Experimental)")
    st.write(
        "This LSTM model is an **experimental neural network** for sequence prediction. "
        "It is compared against classical models and **does not claim to be superior**."
    )

    if st.button("Run LSTM Experiment"):
        with st.spinner("Training LSTM model..."):
            # Load data
            df = load_data(DATASETS[instrument])
            close_values = df[["close"]].values

            # Time-series safe scaling (fit on training part only)
            scaled_close, scaler = scale_close_series(close_values, train_ratio=0.8)

            # Create sequences (samples, window, 1)
            X, y = create_sequences(scaled_close, window_size)

            # Train-test split (time-based)
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Build model
            model = build_lstm_model((X_train.shape[1], 1))

            # Early stopping for better generalization (bias–variance control)
            early_stop = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            # Train
            model.fit(
                X_train,
                y_train,
                validation_split=0.1,
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0,
            )

            # Evaluate
            preds = model.predict(X_test)
            preds_inv = scaler.inverse_transform(preds)
            y_test_2d = y_test.reshape(-1, 1)
            y_test_inv = scaler.inverse_transform(y_test_2d)

            rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))

            # Predict next day
            last_seq = scaled_close[-window_size:]
            last_seq = last_seq.reshape(1, window_size, 1)
            next_pred_scaled = model.predict(last_seq)
            next_pred = scaler.inverse_transform(next_pred_scaled)

        # ---------------- RESULTS ----------------
        st.success("LSTM experiment complete")

        st.metric(
            label="Predicted Next Day Close (LSTM)",
            value=round(float(next_pred[0][0]), 2),
        )

        st.write("**Instrument:**", instrument)
        st.write("**Model (Neural Network):** LSTM (TensorFlow/Keras)")
        st.write("**RMSE (Test Set):**", round(float(rmse), 4))
        st.write("**Last Available Date:**", df.iloc[-1]["datetime"].date())

        st.caption(
            "⚠️ This is an **experimental deep learning model** for academic demonstration only. "
            "It should **not** be used for trading or financial advice."
        )
