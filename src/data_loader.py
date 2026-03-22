import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f'Usunięto {n_dupes} duplikatów')
        df = df.drop_duplicates()

    df['Class'] = df['Class'].map({'Control': 0, 'ADHD': 1})

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f'{missing} brakujacych wartosci - uzupelnianie mediana')
        df = df.fillna(df.median(numeric_only=True))

    return df

def bandpass_filter(signal, fs=128, lowcut=0.5, highcut=40.0, order=4):
    """Filtr pasmowy - usuwa artefakty mięśniowe (>40Hz) i drift (<0.5Hz)"""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=-1)


def segment_into_epochs(df: pd.DataFrame, epoch_length: int = 256, fs: int = 128) -> list:
    eeg_channels = [col for col in df.columns if col not in ['ID', 'Class']]
    epochs = []

    for patient_id, group in df.groupby('ID'):
        label = group['Class'].iloc[0]
        signal = group[eeg_channels].values

        n_epochs = len(signal) // epoch_length
        for i in range(n_epochs):
            start = i * epoch_length
            end = start + epoch_length
            epoch = signal[start:end].T  # (n_channels, epoch_length)

            # filtr pasmowy 0.5-40 Hz - usuwa artefakty
            epoch = bandpass_filter(epoch, fs=fs)

            epochs.append({
                'signal': epoch,
                'label': label,
                'patient_id': patient_id
            })

    return epochs

