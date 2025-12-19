import os

from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd

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

app = Flask(__name__)

# ---------------- DATASET REGISTRY ----------------
DATASETS = {
    "BankNifty": "data/BANKNIFTY_FUT.csv",
    "Nifty": "data/NIFTY_FUT.csv",
    "Gold": "data/GOLD_FUT.csv",
    "Silver": "data/SILVER_FUT.csv",
    "Crude Oil": "data/CRUDEOIL_FUT.csv",
    "Natural Gas": "data/NATURALGAS_FUT.csv",
}

SHOWCASE_FILES = {
    "Combined": "Combined_Showcase.html",
    "BankNifty": "BANKNIFTY_Showcase.html",
    "Nifty": "NIFTY_Showcase.html",
    "Gold": "GOLD_Showcase.html",
    "Silver": "SILVER_Showcase.html",
    "Crude Oil": "CRUDEOIL_Showcase.html",
    "Natural Gas": "NATURALGAS_Showcase.html",
}


# ---------------- HELPER FUNCTIONS (ML CORE) ----------------
def load_data(path: str) -> pd.DataFrame:
    """Load and sort EOD data. Only datetime and close are used for prediction."""
    full_path = os.path.join(app.root_path, path)
    df = pd.read_csv(full_path)
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


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    """Landing page: project overview and navigation."""
    return render_template("index.html")


@app.route("/showcase")
def showcase():
    """
    Showcase page: lets user pick an instrument and view the corresponding
    HTML notebook exported from Jupyter (EDA & comparison).
    """
    selected = request.args.get("instrument", "Combined")
    if selected not in SHOWCASE_FILES:
        selected = "Combined"

    return render_template(
        "showcase.html",
        instruments=list(SHOWCASE_FILES.keys()),
        selected_instrument=selected,
        showcase_files=SHOWCASE_FILES,
    )


@app.route("/showcase/file/<path:filename>")
def showcase_file(filename: str):
    """Serve the static HTML notebook files from data/notebooks."""
    notebooks_dir = os.path.join(app.root_path, "data", "notebooks")
    return send_from_directory(notebooks_dir, filename)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Prediction page: user selects instrument, model type (classical or LSTM),
    and window size; app returns next-day EOD close and evaluation metrics.
    """
    instruments = list(DATASETS.keys())
    classical_models = [
        "Linear Regression",
        "Polynomial Regression (Degree 2)",
        "Random Forest Regressor",
    ]

    context = {
        "instruments": instruments,
        "classical_models": classical_models,
        "result": None,
        "error": None,
    }

    if request.method == "POST":
        try:
            instrument = request.form.get("instrument")
            model_family = request.form.get("model_family")  # "classical" or "lstm"
            window_size = int(request.form.get("window_size", "10"))
            classical_model_name = request.form.get("classical_model")

            if instrument not in DATASETS:
                raise ValueError("Invalid instrument selected.")

            df = load_data(DATASETS[instrument])
            close_values = df[["close"]].values

            # Time-series safe scaling
            scaled_close, scaler = scale_close_series(close_values, train_ratio=0.8)

            # Supervised sequences
            X, y = create_sequences(scaled_close, window_size)

            # Time-ordered split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if model_family == "classical":
                if classical_model_name not in classical_models:
                    raise ValueError("Invalid classical model selected.")

                # Flatten for scikit-learn (samples, window)
                X_train_flat = X_train.reshape(X_train.shape[0], X_train.shape[1])
                X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])

                model = build_classical_model(classical_model_name)
                model.fit(X_train_flat, y_train.ravel())

                preds_scaled = model.predict(X_test_flat)
                preds_scaled_2d = preds_scaled.reshape(-1, 1)
                y_test_2d = y_test.reshape(-1, 1)

                preds = scaler.inverse_transform(preds_scaled_2d).ravel()
                y_true = scaler.inverse_transform(y_test_2d).ravel()

                mae = mean_absolute_error(y_true, preds)
                mse = mean_squared_error(y_true, preds)
                rmse = float(np.sqrt(mse))
                r2 = r2_score(y_true, preds)

                # Next-day prediction
                last_seq = scaled_close[-window_size:].reshape(1, window_size)
                next_scaled = model.predict(last_seq)
                next_price = scaler.inverse_transform(
                    np.array(next_scaled).reshape(-1, 1)
                )[0, 0]

                context["result"] = {
                    "instrument": instrument,
                    "model_type": "Classical Regression",
                    "model_name": classical_model_name,
                    "predicted_price": float(next_price),
                    "mae": float(mae),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "r2": float(r2),
                    "last_date": df.iloc[-1]["datetime"].date(),
                }

            else:  # LSTM experimental
                # LSTM expects shape (samples, window, 1)
                model = build_lstm_model((X_train.shape[1], 1))

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True,
                )

                model.fit(
                    X_train,
                    y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=0,
                )

                preds = model.predict(X_test)
                preds_inv = scaler.inverse_transform(preds)
                y_test_2d = y_test.reshape(-1, 1)
                y_test_inv = scaler.inverse_transform(y_test_2d)

                rmse = float(
                    np.sqrt(mean_squared_error(y_test_inv, preds_inv))
                )

                # Next-day prediction
                last_seq = scaled_close[-window_size:]
                last_seq = last_seq.reshape(1, window_size, 1)
                next_pred_scaled = model.predict(last_seq)
                next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]

                context["result"] = {
                    "instrument": instrument,
                    "model_type": "Neural Network (LSTM – Experimental)",
                    "model_name": "LSTM",
                    "predicted_price": float(next_pred),
                    "rmse": float(rmse),
                    "last_date": df.iloc[-1]["datetime"].date(),
                }

        except Exception as exc:  # noqa: BLE001
            context["error"] = str(exc)

    return render_template("predict.html", **context)


if __name__ == "__main__":
    # Run in development mode. For deployment, use a production WSGI server.
    app.run(debug=True)



