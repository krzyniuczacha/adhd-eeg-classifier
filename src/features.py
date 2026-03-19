import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def extract_stat_features(signal: np.ndarray) -> dict:
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'variance': np.var(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'peak_to_peak': np.max(signal) - np.min(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'zero_crossings': np.sum(np.diff(np.sign(signal)) != 0),
    }

def extract_freq_features(signal: np.ndarray, fs: int = 128) -> dict:
    freqs, psd = welch(signal, fs=fs, nperseg= min(256, len(signal)))

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50),
    }

    features = {}
    total_power = np.trapz(psd, freqs)

    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])

        features[f'{band_name}_power'] = band_power
        features[f'{band_name}_relative'] = band_power / total_power

    features['theta_beta_ratio'] = features['theta_power'] / (features['beta_power'] + 1e-10)

    return features

def extract_all_features(df: pd.DataFrame, fs: int = 128) -> pd.DataFrame:
    eeg_channels = [col for col in df.colums if col not in ['ID', 'Class']]
    all_features = []

    for patient_id, group in df.groupby('ID'):
        patient_features = {'ID': patient_id, 'Class': group['Class'].iloc[0]}

        for channel in eeg_channels:
            signal = group[channel].values

            stat_features = extract_stat_features(signal)
            for feature_name, value in stat_features.items():
                patient_features[f'{channel}_{feature_name}'] = value


            freq_features = extract_freq_features(signal, fs)
            for feature_name, value in freq_features.items():
                patient_features[f'{channel}_{feature_name}'] = value

        all_features.append(patient_features)

    return pd.DataFrame(all_features)
