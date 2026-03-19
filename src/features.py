import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import antropy as ant

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
    features['theta_alpha_ratio'] = features['theta_power'] / (features['alpha_power'] + 1e-10)
    features['alpha_beta_ratio'] = features['alpha_power'] / (features['beta_power'] + 1e-10)

    log_freqs = np.log10(freqs[1:] + 1e-10)
    log_psd = np.log10(psd[1:] + 1e-10)
    features['spectral-slope'] = np.polyfit(log_freqs, log_psd, 1)[0]

    psd_norm = psd / (np.sum(psd) + 1e-10)
    features['spectral-entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    return features

def extract_nonlinear_features(signal: np.ndarray) -> dict:
    features = {}

    features['sample_entropy'] = ant.sample_entropy(signal) # mierzy nieprzewidywalnosc sygnalu
    features['perm_entropy'] = ant.perm_entropy(signal, normalize=True) # mierzy zlozonosc porzadku warotsci
    features['approx_entropy'] = ant.approx_entropy(signal) # podobna do probkowej ale szybsza
    features['higuchi_fd'] = ant.higuchi_fd(signal) # wymiar fraktalny Higuchiego - chropowatosc sygnalu
    features['katz_fd'] = ant.katz_fd(signal)  # wymiar fraktalny Katza
    features['dfa'] = ant.detrained_fluaction(signal) # dfa - dlugozasiegowe korelacje

    return features


def extract_cross_channel_features(epoch: np.ndarray, fs: int = 128) -> dict:
    n_channels = epoch.shape[0]
    features = {}

    corr_matrix = np.corrcoef(epoch)
    upper_triangle = corr_matrix[np.triu_indices(n_channels, k=1)]

    features['mean_correlation'] = np.mean(upper_triangle)
    features['std_correlation'] = np.std(upper_triangle)
    features['min_correlation'] = np.min(upper_triangle)
    features['max_correlation'] = np.max(upper_triangle)

    # Ansymetria miedzypolkolowa
    left_channels = [0, 2, 4, 6, 8, 10, 12, 14] # Fp1, F3, C3, P3, O1, F7, T7, P7
    right_channels = [1, 3, 5, 7, 9, 11, 13, 15] # Fp2, F4, C4, P4, O2, F8, T8, P8

    for i, (l,r) in enumerate(zip(left_channels, right_channels)):
        left_power = np.var(epoch[l])
        right_power = np.var(epoch[r])
        # Asymetria = (r - l) / (r + l)
        features[f'assymetry_pair_{i}'] = (right_power - left_power) / (right_power + left_power + 1e-10)

    return features

def extract_all_features_epoch_based(epochs: list, fs: int = 128) -> tuple:
    all_features = []
    labels = []
    groups = []

    eeg_channel_names = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
                         'O1','O2','F7','F8','T7','T8','P7','P8','Fz','Cz','Pz']

    for i, epoch_data in enumerate(epochs):
        epoch = epoch_data['signal']
        features = {}

        for ch_idx, ch_name in enumerate(eeg_channel_names):
            signal = epoch[ch_idx]

            stat_features = extract_stat_features(signal)
            for k, v in stat_features.items():
                features[f'{ch_name}_{k}'] = v

            freq_features = extract_freq_features(signal, fs)
            for k, v in freq_features.items():
                features[f'{ch_name}_{k}'] = v

            nonlin_features = extract_nonlinear_features(signal)
            for k, v in nonlin_features.items():
                features[f'{ch_name}_{k}'] = v

        cross_features = extract_cross_channel_features(epoch, fs)
        features.update(cross_features)

        all_features.append(features)
        labels.append(epoch_data['label'])
        groups.append(epoch_data['patient_id'])

        X = pd.DataFrame(all_features).values
        y = np.array(labels)
        groups = np.array(groups)

        return X, y, groups