import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.fillna(df.median(numeric_only=True))

    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    eeg_channels = [col for col in df.columns if col not in ['ID', 'Class']]
    X = df[eeg_channels]
    y = df[['Class']].values
    groups = df['ID'].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_patients = set(groups[train_idx])
    test_patients = set(groups[test_idx])
    assert len(train_patients & test_patients) == 0, "Subject leak"

    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


