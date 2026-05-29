"""
ml_model.py - Machine Learning logic using XGBoost for score prediction.

Handles feature engineering (sliding window, rolling mean), model training,
and next-score prediction.
"""

import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


# --- Constants ---
WINDOW_SIZE = 3  # Number of previous scores used as features
MIN_ROUNDS_REQUIRED = 5  # Minimum data points needed before training


def build_features(scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y from a chronological list of scores.

    Features (sliding window of length WINDOW_SIZE):
        - prev_score_1 : t-1
        - prev_score_2 : t-2
        - prev_score_3 : t-3
        - rolling_mean_3 : mean of t-1, t-2, t-3

    Parameters
    ----------
    scores : List[float]
        Chronologically ordered list of multiplier scores.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X (feature matrix) and y (target vector) ready for model training.
    """
    if len(scores) < MIN_ROUNDS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_ROUNDS_REQUIRED} rounds of data to train. "
            f"Currently have {len(scores)}."
        )

    df = pd.DataFrame({"score": scores})

    # --- Sliding window features ---
    for lag in range(1, WINDOW_SIZE + 1):
        df[f"prev_score_{lag}"] = df["score"].shift(lag)

    # --- Rolling mean feature ---
    df["rolling_mean_3"] = (
        df["score"].shift(1).rolling(window=WINDOW_SIZE).mean()
    )

    # Drop rows with NaN created by shifting
    df_clean = df.dropna().reset_index(drop=True)

    feature_cols = [
        "prev_score_1",
        "prev_score_2",
        "prev_score_3",
        "rolling_mean_3",
    ]

    X = df_clean[feature_cols].values
    y = df_clean["score"].values

    return X, y


def train_model(
    X: np.ndarray, y: np.ndarray
) -> Tuple[XGBRegressor, Optional[float]]:
    """
    Train an XGBoost regressor on the provided features and target.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.

    Returns
    -------
    Tuple[XGBRegressor, Optional[float]]
        The trained model and the root mean squared error (RMSE) on the
        training set. If there is only one sample, MSE will be None.
    """
    model = XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # Calculate training error
    y_pred = model.predict(X)
    mse: Optional[float] = None
    if len(y) > 1:
        mse = mean_squared_error(y, y_pred)

    return model, mse


def predict_next(
    model: XGBRegressor, recent_scores: List[float]
) -> float:
    """
    Predict the next score using the most recent WINDOW_SIZE scores.

    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    recent_scores : List[float]
        The last few scores from the database (at least WINDOW_SIZE).

    Returns
    -------
    float
        Predicted next score, capped at a minimum of 1.00 and rounded
        to 2 decimal places.
    """
    if len(recent_scores) < WINDOW_SIZE:
        # Pad with the last available value if not enough history
        pad_needed = WINDOW_SIZE - len(recent_scores)
        recent_scores = [recent_scores[0]] * pad_needed + recent_scores

    last_3 = recent_scores[-WINDOW_SIZE:]
    rolling_mean = np.mean(last_3)

    features = np.array(
        [[last_3[2], last_3[1], last_3[0], rolling_mean]]
        # Note: prev_score_1 = most recent single lag (t-1),
        #       prev_score_2 = t-2,
        #       prev_score_3 = t-3
    )

    predicted = model.predict(features)[0]

    # Cap minimum at 1.00 and round to 2 decimal places
    predicted = max(round(float(predicted), 2), 1.00)

    return predicted


def train_and_predict_from_history(
    scores: List[float],
) -> Tuple[float, Optional[float], int]:
    """
    High-level convenience function: build features, train model, predict next.

    Parameters
    ----------
    scores : List[float]
        All historical multiplier scores in chronological order.

    Returns
    -------
    Tuple[float, Optional[float], int]
        (predicted_next_score, training_mse, total_samples_used)

    Raises
    ------
    ValueError
        If there are fewer than MIN_ROUNDS_REQUIRED scores.
    """
    if len(scores) < MIN_ROUNDS_REQUIRED:
        raise ValueError(
            f"Please enter at least {MIN_ROUNDS_REQUIRED} rounds of data "
            f"to start training. Currently have {len(scores)}."
        )

    X, y = build_features(scores)
    model, mse = train_model(X, y)
    prediction = predict_next(model, scores)

    return prediction, mse, len(scores)


if __name__ == "__main__":
    # Quick test with sample data
    test_scores = [1.23, 5.40, 2.10, 1.05, 12.30, 3.50, 2.80, 1.90]
    try:
        pred, mse, n = train_and_predict_from_history(test_scores)
        print(f"Test prediction: {pred} (MSE: {mse:.6f}, samples: {n})")
    except ValueError as e:
        print(f"Error: {e}")