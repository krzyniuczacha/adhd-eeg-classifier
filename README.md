# ADHD Classification Based on EEG Signals

A binary classifier for diagnosing ADHD based on multichannel EEG signals (19 channels, 10-20 system).

Project developed as a recruitment task for the AI section of KN Neuron (Spring 2026).

## Project Structure

```
adhd-eeg-classifier/
├── data/
│   └── adhdata.csv              # EEG data (Kaggle)
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis (EDA)
├── src/
│   ├── __init__.py              # Python package marker
│   ├── data_loader.py           # CSV loading, bandpass filtering, epoch segmentation
│   ├── features.py              # feature extraction (statistical, frequency, Hjorth, nonlinear, cross-channel)
│   ├── model.py                 # model definitions (RF, SVM, CNN)
│   ├── train.py                 # training, cross-validation, data augmentation
│   └── evaluate.py              # metrics, confusion matrix, ROC, model comparison
├── results/                     # plots and saved models (auto-generated)
├── main.py                      # main pipeline
├── requirements.txt
└── README.md
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/krzyniuczacha/adhd-eeg-classifier.git
cd adhd-eeg-classifier

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Download from https://www.kaggle.com/datasets/danizo/eeg-dataset-for-adhd/data
# and place adhdata.csv in the data/ folder

# 5. Run the pipeline
python main.py
```

Results (plots, confusion matrices, ROC curves) will be saved in the `results/` folder.

## Pipeline

### 1. Data Loading & Preprocessing

- Duplicate removal, missing value imputation with median
- **Butterworth bandpass filter** (0.5–40 Hz, order 4) — removes drift (<0.5 Hz) and muscle artifacts (>40 Hz)
- **Epoch segmentation** — 2-second windows (256 samples at fs=128 Hz)

### 2. Feature Extraction

For each epoch and each of the 19 EEG channels, features are computed across four categories:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Statistical** | mean, std, variance, skew, kurtosis, min, max, peak-to-peak, RMS, zero crossings | Basic description of the signal amplitude distribution |
| **Frequency** (Welch PSD) | Absolute and relative power in delta/theta/alpha/beta/gamma bands, theta/beta ratio, theta/alpha ratio, spectral slope, spectral entropy | Theta/beta ratio is a key ADHD biomarker — elevated theta and reduced beta activity |
| **Hjorth** | Activity, Mobility, Complexity | Classic EEG features: signal power, dominant frequency, frequency variability |
| **Nonlinear** (antropy) | Sample entropy, permutation entropy, approximate entropy, Higuchi FD, Katz FD, DFA | Signal complexity and regularity — ADHD is associated with altered nonlinear dynamics |

Additional **cross-channel** features:
- Correlation matrix statistics (mean, std, min, max)
- **Interhemispheric asymmetry** — overall and per band (theta/alpha) for 8 electrode pairs (Fp1-Fp2, F3-F4, C3-C4, P3-P4, O1-O2, F7-F8, T7-T8, P7-P8)

**Feature selection:** `SelectFromModel` with Random Forest (threshold: 75th percentile of importance) — reduces ~690 features to ~170 most relevant ones.

### 3. Data Split

**Subject Split** (`GroupShuffleSplit`, 80/20) — data from the same patient **never** appears in both training and test sets. This prevents data leakage and ensures a reliable evaluation of model generalization.

### 4. Models

Comparison of four approaches:

**Random Forest** (scikit-learn)
- 1000 trees, `class_weight='balanced'`, OOB score
- Cross-validation: `StratifiedGroupKFold` (5 folds with patient groups)

**SVM with RBF kernel** (scikit-learn)
- `C=10.0, gamma=0.01`, balanced class weights
- Cross-validation as above

**1D CNN** (PyTorch)
- 3 convolutional blocks (32→64→64 filters) on raw EEG signals (19×256)
- Wide filters in the first layer (kernel=15) capture slow delta/theta oscillations
- Augmentation: Gaussian noise, time shifts, amplitude scaling, channel dropout
- Label smoothing, BCEWithLogitsLoss with class weights, AdamW + CosineAnnealingWarmRestarts
- Early stopping with patience=15

**Ensemble** (Soft Voting)
- Weighted average of probabilities: 45% RF + 30% SVM + 25% CNN
- Weights proportional to individual model performance

### 5. Evaluation

- **Metrics:** Accuracy, Precision, Recall, F1-score, Classification Report
- **Visualizations:** Confusion Matrix, ROC curve with AUC, model comparison chart, feature importance
- **Patient-level aggregation** — averaging epoch probabilities per patient, threshold at 0.5

## Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Random Forest | 0.88 | 0.92 | 0.88 | 0.90 |
| SVM (RBF) | 0.84 | 0.90 | 0.84 | 0.87 |
| CNN | 0.82 | 0.81 | 0.91 | 0.86 |
| Ensemble | **0.89** | **0.92** | **0.89** | **0.90** |

Patient-level prediction (Ensemble): **accuracy 0.88**, F1 0.89 on 25 test patients.

![Model Comparison](results/model_comparison.png)

### Conclusions

- **Random Forest** achieved the best results among individual models (F1=0.90). Hand-crafted features (theta/beta ratio, entropy, asymmetry) proved more effective than raw signals fed to CNN.
- **SVM** performs slightly worse than RF — likely due to the high number of features (>150), where decision trees handle feature spaces better than an RBF kernel.
- **CNN** has the highest recall (0.91) — it misses fewer ADHD cases — but at the cost of precision. Overfitting remains a challenge with a small dataset (~6700 training epochs). Augmentation (noise, time shift, channel dropout) mitigated the issue but did not fully resolve it.
- **Ensemble** combines the strengths of all models and achieves the best overall balance of metrics. CNN's high recall compensates for the conservativeness of RF/SVM.
- **Most important features** are theta/beta ratio, spectral entropy, and interhemispheric asymmetry in the theta band — consistent with clinical ADHD literature.
- Cross-validation with patient groups shows high variance across folds (acc 0.70–0.82), suggesting significant inter-subject variability in EEG signals.

## Technologies

- Python 3.13
- PyTorch (CNN)
- scikit-learn (Random Forest, SVM, preprocessing)
- SciPy (signal filtering, frequency analysis)
- antropy (nonlinear EEG features)
- matplotlib / seaborn (visualizations)
- Jupyter Notebook (EDA)
