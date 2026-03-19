import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f'usuwam {n_dupes} duplikatow')
        df = df.drop_duplicates()

    df['Class'] = df['Class'].map({'Control': 0, 'ADHD': 1})

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f'{missing} brakujacych wartosci - uzupelnianie mediana')
        df = df.fillna(df.median(numeric_only=True))

    return df

def segment_into_epochs(df: pd.DataFrame, epoch_length: int = 256) -> list:
    eeg_channels = [col for col in df.columns if col not in ['ID', 'Class']]
    epochs = []

    for patient_id, group in df.groupby('ID'):
        label = group['Class'].iloc[0]
        signal = group[eeg_channels].values

    n_epochs = len(signal) // epoch_length
    for i in range (n_epochs):
        start = i * epoch_length
        end = start + epoch_length
        epoch = signal[start:end].T

        epochs.append({
            'signal': epoch,
            'label': label,
            'patient_id': patient_id
        })

    return epochs

def split_data_by_subject(epochs: list, test_size: float = 0.2, random_state: int = 42):
    X = np.array([e['signal'] for e in epochs])
    y = np.array([e['label'] for e in epochs])
    groups = np.array([e['patient_id'] for e in epochs])

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_patients = set(groups[train_idx])
    test_patients = set(groups[test_idx])
    assert len(train_patients & test_patients) == 0, "Subject Leak"

    return X_train, X_test, y_train, y_test, groups[train_idx]

def normalize_data(X_train, X_test):
    n_channels = X_train.shape[1]
    scalers = []

    for ch in range(n_channels):
        scaler = StandardScaler()
        X_train[:, ch, :] = scaler.fit_transform(X_train[:, ch, :])
        X_test[:, ch, :] = scaler.fit_transform(X_test[:, ch, :])
        scalers.append(scaler)

    return X_train, X_test, scalers


