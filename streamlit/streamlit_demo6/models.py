import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Isolates strictly numerical sensor data from labels and time."""
    exclude_cols = ['datetime', 'anomaly', 'changepoint']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return df[feature_cols]


def train_predict_zscore(df: pd.DataFrame, training_cutoff: int, z_threshold: float = 4.0) -> pd.DataFrame:
    """Calculates a global Z-Score strictly from the training baseline with a noise floor."""
    df_out = df.copy()
    features = get_sensor_features(df_out)

    # 1. SPLIT
    train_features = features.iloc[:training_cutoff]

    # 2. TRAIN: Calculate global mean and standard deviation on the baseline
    baseline_mean = train_features.mean()
    # Apply a noise floor (0.05) to prevent division by zero on perfectly flat sensors
    baseline_std = train_features.std().clip(lower=0.05)

    # 3. PREDICT
    z_scores = np.abs((features - baseline_mean) / baseline_std)
    df_out['anomaly_score'] = z_scores.max(axis=1)
    df_out['predicted_anomaly'] = (df_out['anomaly_score'] > z_threshold).astype(int)

    return df_out


def train_predict_isolation_forest(df: pd.DataFrame, training_cutoff: int, contamination: float = 0.01) -> pd.DataFrame:
    """Restored to the original, highly sensitive statistical thresholding."""
    df_out = df.copy()
    features = get_sensor_features(df_out)

    # 1. SPLIT
    train_features = features.iloc[:training_cutoff]

    # 2. SCALE (With basic noise floor)
    baseline_mean = train_features.mean()
    baseline_std = train_features.std().clip(lower=0.05)

    scaled_train = (train_features - baseline_mean) / baseline_std
    scaled_all = (features - baseline_mean) / baseline_std

    # 3. TRAIN
    model = IsolationForest(random_state=42)
    model.fit(scaled_train)

    # 4. THRESHOLD (Direct mapping from the slider)
    train_scores = model.decision_function(scaled_train)
    threshold = np.quantile(train_scores, contamination)

    # 5. PREDICT
    all_scores = model.decision_function(scaled_all)
    df_out['predicted_anomaly'] = (all_scores < threshold).astype(int)

    # Invert for UI plotting purposes
    df_out['anomaly_score'] = all_scores * -1

    return df_out


def train_predict_pca(df: pd.DataFrame, training_cutoff: int, n_components: int = 2,
                      contamination: float = 0.01) -> pd.DataFrame:
    """Trains PCA with manual scaling and an enforced minimum reconstruction error threshold."""
    df_out = df.copy()
    features = get_sensor_features(df_out)

    # 1. SPLIT
    train_features = features.iloc[:training_cutoff]

    # 2. SCALE WITH NOISE FLOOR (Replacing StandardScaler)
    baseline_mean = train_features.mean()
    baseline_std = train_features.std().clip(lower=0.05)

    scaled_train = (train_features - baseline_mean) / baseline_std
    scaled_all = (features - baseline_mean) / baseline_std

    # 3. TRAIN
    pca = PCA(n_components=n_components, random_state=42)
    pca_transformed_train = pca.fit_transform(scaled_train)
    train_reconstructed = pca.inverse_transform(pca_transformed_train)
    train_mse = np.mean(np.power(scaled_train - train_reconstructed, 2), axis=1)

    # 4. THRESHOLD
    base_threshold = np.quantile(train_mse, 1 - contamination)
    error_threshold = max(base_threshold, 0.25)

    # 5. PREDICT
    pca_transformed_all = pca.transform(scaled_all)
    all_reconstructed = pca.inverse_transform(pca_transformed_all)
    all_mse = np.mean(np.power(scaled_all - all_reconstructed, 2), axis=1)

    df_out['anomaly_score'] = all_mse
    df_out['predicted_anomaly'] = (df_out['anomaly_score'] > error_threshold).astype(int)

    return df_out