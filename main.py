import numpy as np
import pandas as pd
from src.data_loader import load_data, segment_into_epochs
from src.features import extract_all_features_epoch_based
from src.model import create_rf, create_svm, save_model
from src.train import cross_validate, train_final_model, train_cnn
from src.evaluate import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, plot_comparison, aggregate_patient_predictions
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SAMPLING_RATE = 128
EPOCH_LENGTH = 256  # 2s


def main():
    np.random.seed(42)
    df = load_data("data/adhdata.csv")
    epochs = segment_into_epochs(df, epoch_length=EPOCH_LENGTH, fs=SAMPLING_RATE)

    # --- Ekstrakcja cech (dla RF i SVM) ---
    X, y, groups, feature_names = extract_all_features_epoch_based(epochs, fs=SAMPLING_RATE)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # surowe sygnaly (dla CNN)
    X_raw = np.array([e['signal'] for e in epochs])  # (n_epochs, 19, 256)

    print(f"\n--- Rozklad klas ---")
    print(f"Wszystkie: Zdrowy={sum(y==0)}, ADHD={sum(y==1)}, razem={len(y)}")
    print(f"Liczba cech: {X.shape[1]}")
    print(f"Surowe sygnaly: {X_raw.shape}")

    # --- Feature selection (dla RF i SVM) ---
    print("\n=== Selekcja cech ===")
    scaler_fs = StandardScaler()
    X_scaled = scaler_fs.fit_transform(X)

    selector_model = RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced'
    )
    selector_model.fit(X_scaled, y)

    importances = selector_model.feature_importances_
    threshold = np.percentile(importances, 75)
    selector = SelectFromModel(selector_model, threshold=threshold)
    selector.fit(X_scaled, y)

    selected_mask = selector.get_support()
    selected_names = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    X_selected = X[:, selected_mask]
    print(f"Wybrano {X_selected.shape[1]} z {X.shape[1]} cech")

    # --- Wspolny podzial train/test (subject split) ---
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_selected, y, groups))

    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]

    # surowe sygnaly - ten sam podzial
    X_raw_train, X_raw_test = X_raw[train_idx], X_raw[test_idx]

    print(f"\nTrain: Zdrowy={sum(y_train==0)}, ADHD={sum(y_train==1)}")
    print(f"Test:  Zdrowy={sum(y_test==0)}, ADHD={sum(y_test==1)}")

    all_results = {}

    # 1. RANDOM FOREST
    print("\n" + "="*50)
    print("1. RANDOM FOREST")
    print("="*50)

    print("\n--- Cross-Validation ---")
    rf = create_rf()
    cross_validate(rf, X_selected, y, groups)

    rf = create_rf()
    rf, scaler_rf = train_final_model(rf, X_train, y_train)
    save_model(rf, scaler_rf, selector)
    print(f"OOB score: {rf.oob_score_:.4f}")

    X_test_rf = scaler_rf.transform(X_test)
    rf_preds = rf.predict(X_test_rf)
    rf_probs = rf.predict_proba(X_test_rf)[:, 1]

    print("\n--- Wyniki (epoki) ---")
    all_results['Random Forest'] = evaluate_model(y_test, rf_preds)
    plot_confusion_matrix(y_test, rf_preds, model_name="RF")
    plot_roc_curve(y_test, rf_probs, model_name="RF")
    plot_feature_importance(rf, selected_names, top_n=20)

    # 2. SVM
    print("\n" + "="*50)
    print("2. SVM (RBF)")
    print("="*50)

    print("\n--- Cross-Validation ---")
    svm = create_svm()
    cross_validate(svm, X_selected, y, groups)

    svm = create_svm()
    svm, scaler_svm = train_final_model(svm, X_train, y_train)

    X_test_svm = scaler_svm.transform(X_test)
    svm_preds = svm.predict(X_test_svm)
    svm_probs = svm.predict_proba(X_test_svm)[:, 1]

    print("\n--- Wyniki (epoki) ---")
    all_results['SVM (RBF)'] = evaluate_model(y_test, svm_preds)
    plot_confusion_matrix(y_test, svm_preds, model_name="SVM")
    plot_roc_curve(y_test, svm_probs, model_name="SVM")

    # 3. CNN (PyTorch)
    print("\n" + "="*50)
    print("3. CNN (PyTorch)")
    print("="*50)

    cnn_model, cnn_preds, cnn_probs, cnn_norm = train_cnn(
        X_raw_train, y_train, X_raw_test, y_test,
        n_channels=19, n_samples=EPOCH_LENGTH,
    )

    print("\n--- Wyniki (epoki) ---")
    all_results['CNN'] = evaluate_model(y_test, cnn_preds)
    plot_confusion_matrix(y_test, cnn_preds, model_name="CNN")
    plot_roc_curve(y_test, cnn_probs, model_name="CNN")


    # 4. ENSEMBLE (Soft Voting: RF + SVM + CNN)
    print("\n" + "="*50)
    print("4. ENSEMBLE (Soft Voting)")
    print("="*50)

    # srednia waziona prawdopodobienstw (RF najlepszy -> wyzszy weight)
    ensemble_probs = 0.45 * rf_probs + 0.30 * svm_probs + 0.25 * cnn_probs
    ensemble_preds = (ensemble_probs > 0.5).astype(int)

    print("\n--- Wyniki (epoki) ---")
    all_results['Ensemble'] = evaluate_model(y_test, ensemble_preds)
    plot_confusion_matrix(y_test, ensemble_preds, model_name="Ensemble")
    plot_roc_curve(y_test, ensemble_probs, model_name="Ensemble")

    # POROWNANIE MODELI
    plot_comparison(all_results)

    # --- Agregacja na poziomie pacjenta (ensemble) ---
    print("\n=== Predykcja na poziomie pacjenta (Ensemble) ===")
    y_patient_true, y_patient_pred = aggregate_patient_predictions(
        ensemble_probs, y_test, groups_test
    )
    print(f"Liczba pacjentow w tescie: {len(np.unique(groups_test))}")
    evaluate_model(y_patient_true, y_patient_pred)


if __name__ == '__main__':
    main()
