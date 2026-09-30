"""LSTM forecasters (Keras / TensorFlow).

TensorFlow is imported inside the functions so the baselines, tests and
data pipeline run without it.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .features import make_windows, scaled_return_windows


def _keras():
    from tensorflow import keras
    return keras


def build_return_lstm(window, n_features, units=32, dropout=0.2, lr=1e-3):
    """Compact LSTM that forecasts the next-day (standardised) log return."""
    keras = _keras()
    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        keras.layers.LSTM(units),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1),
    ])
    # Huber loss limits the pull of crash/rally days on the fit
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.Huber())
    return model


def build_original_lstm(window):
    """The 2023 architecture: 4 stacked ReLU LSTMs (50/60/80/120) with rising dropout."""
    keras = _keras()
    layers = keras.layers
    model = keras.Sequential([
        keras.Input(shape=(window, 1)),
        layers.LSTM(50, activation="relu", return_sequences=True), layers.Dropout(0.2),
        layers.LSTM(60, activation="relu", return_sequences=True), layers.Dropout(0.3),
        layers.LSTM(80, activation="relu", return_sequences=True), layers.Dropout(0.4),
        layers.LSTM(120, activation="relu"), layers.Dropout(0.5),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _fit(model, X, y, epochs, batch_size, patience):
    keras = _keras()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True)
    # validation_split takes the LAST 15% of samples, i.e. the most recent
    # training period, so validation never looks at data older than training
    model.fit(X, y, validation_split=0.15, epochs=epochs, batch_size=batch_size,
              callbacks=[stop], verbose=0)
    return model


def return_lstm(feat, window, train_pos, test_pos, seed=0, n_seeds=3,
                epochs=100, batch_size=64, patience=10):
    """Forecast next-day log returns; averages an ensemble of ``n_seeds`` runs."""
    keras = _keras()
    X_tr, y_tr, X_te = scaled_return_windows(feat, window, train_pos, test_pos)
    mu, sd = y_tr.mean(), y_tr.std()
    preds = []
    for k in range(n_seeds):
        keras.utils.set_random_seed(seed + k)
        model = build_return_lstm(window, X_tr.shape[2])
        _fit(model, X_tr, (y_tr - mu) / sd, epochs, batch_size, patience)
        preds.append(model.predict(X_te, verbose=0).ravel() * sd + mu)
    return np.mean(preds, axis=0)


def original_lstm(feat, window, train_pos, test_pos, seed=0,
                  epochs=30, batch_size=50, patience=5):
    """The 2023 approach, trained on scaled price LEVELS, with the inverse scaling fixed.

    Returns implied log returns so it is scored exactly like every other model.
    """
    keras = _keras()
    keras.utils.set_random_seed(seed)
    prices = feat["price"].values
    scaler = MinMaxScaler().fit(prices[:test_pos[0], None])  # pre-test rows only
    scaled = scaler.transform(prices[:, None]).ravel()
    X_tr = make_windows(scaled, train_pos, window)
    X_te = make_windows(scaled, test_pos, window)
    model = build_original_lstm(window)
    _fit(model, X_tr, scaled[train_pos], epochs, batch_size, patience)
    pred_scaled = model.predict(X_te, verbose=0).reshape(-1, 1)
    # Correct inverse: x * (max - min) + min. The 2023 notebook multiplied by a
    # hard-coded factor and dropped the "+ min" term.
    pred_price = np.clip(scaler.inverse_transform(pred_scaled).ravel(), 1.0, None)
    return np.log(pred_price / prices[test_pos - 1])
