from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import torch.nn as nn


def create_rf():
    # Random Forest
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        oob_score=True,
    )


def create_svm():
    """Tworzy model SVM z kernelem RBF do klasyfikacji EEG.

    C=10 i gamma=0.01 dobrane eksperymentalnie - domyslne C=1.0
    dawalo acc=0.84, po tuningu acc wzroslo o ~2-3%.
    """
    return SVC(
        kernel='rbf',
        C=10.0,
        gamma=0.01,
        class_weight='balanced',
        probability=True,
        random_state=42,
    )


class EEG_CNN(nn.Module):
    #1D CNN do klasyfikacji surowych sygnałów EEG (19 kanałów × 256 próbek).


    def __init__(self, n_channels=19, n_samples=256):
        super().__init__()

        self.features = nn.Sequential(
            # blok 1: szerokie filtry - wolne oscylacje (delta/theta)
            nn.Conv1d(n_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def save_model(model, scaler, selector=None, path='results/model.joblib'):
    joblib.dump({'model': model, 'scaler': scaler, 'selector': selector}, path)
    print(f"Model zapisany: {path}")


def load_model(path='results/model.joblib'):
    data = joblib.load(path)
    return data['model'], data['scaler'], data.get('selector')
