import numpy as np
from sklearn.metrics import (
accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_patient_predictions(y_pred_epochs, y_true_epochs, patient_ids):
    """Agreguje predykcje epok do poziomu pacjenta (majority voting)."""
    patient_preds = {}
    patient_labels = {}

    for pred, true, pid in zip(y_pred_epochs, y_true_epochs, patient_ids):
        if pid not in patient_preds:
            patient_preds[pid] = []
            patient_labels[pid] = true
        patient_preds[pid].append(pred)

    y_patient_pred = []
    y_patient_true = []

    for pid in patient_preds:
        avg_pred = np.mean(patient_preds[pid])
        y_patient_pred.append(1 if avg_pred > 0.5 else 0)
        y_patient_true.append(patient_labels[pid])

    return np.array(y_patient_true), np.array(y_patient_pred)

def evaluate_model(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    for metric, value in metrics.items():
        print(f" {metric:>12}: {value:.4f}")
    print(f"Classification report: ")
    print(classification_report(y_true, y_pred, target_names=['Zdrowy', 'ADHD']))

    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name=""):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Zdrowy", "ADHD"], yticklabels=["Zdrowy", "ADHD"])
    title = f"Confusion Matrix - {model_name}" if model_name else "Confusion Matrix"
    plt.title(title)
    plt.ylabel("Prawdziwa klasa")
    plt.xlabel("Predykcja modelu")
    plt.tight_layout()
    fname = f"results/confusion_matrix_{model_name}.png" if model_name else "results/confusion_matrix.png"
    plt.savefig(fname, dpi=150)
    plt.show()

def plot_roc_curve(y_true, y_prob, model_name=""):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'(AUC = {roc_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = f"Krzywa ROC - {model_name}" if model_name else "Krzywa ROC"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fname = f"results/roc_curve_{model_name}.png" if model_name else "results/roc_curve.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    return roc_auc

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Waznosc cechy')
    plt.title(f'Top {top_n} najwazniejszych cech')
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150)
    plt.show()


def plot_comparison(results):
    """Tabela i wykres porownujacy wyniki wszystkich modeli"""
    print("\n" + "="*60)
    print("POROWNANIE MODELI")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*60)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    print("="*60)

    # wykres porownawczy
    model_names = list(results.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        values = [results[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_ylabel('Wartosc metryki')
    ax.set_title('Porownanie modeli')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.show()

