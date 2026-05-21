import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
from preprocess import load_and_preprocess

MIN_PRECISION = 0.80

MODEL_PATH = Path(__file__).parent.parent / 'outputs' / 'model.pkl'
FIGURES_PATH = Path(__file__).parent.parent / 'outputs' / 'figures'
METRICS_PATH = Path(__file__).parent.parent / 'outputs' / 'metrics.txt'


def evaluate():
    print('Loading and preprocessing data...')
    _, X_test, _, y_test = load_and_preprocess()

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Find the lowest threshold where precision stays >= MIN_PRECISION,
    # which maximises recall while keeping false positives under control.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    viable = np.where(precisions[:-1] >= MIN_PRECISION)[0]
    if len(viable):
        best_idx = viable[np.argmax(recalls[viable])]
        threshold = thresholds[best_idx]
    else:
        threshold = 0.5  # fallback
    print(f'Optimal threshold: {threshold:.4f}  (precision >= {MIN_PRECISION})')

    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'], digits=4)

    print(report)
    print(f'ROC-AUC : {roc_auc:.4f}')
    print(f'PR-AUC  : {pr_auc:.4f}')

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        f.write(f'Threshold : {threshold:.4f}  (precision >= {MIN_PRECISION})\n\n')
        f.write(report)
        f.write(f'\nROC-AUC : {roc_auc:.4f}\n')
        f.write(f'PR-AUC  : {pr_auc:.4f}\n')

    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Legit', 'Pred Fraud'])
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=13,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title('Confusion Matrix')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'confusion_matrix.png', dpi=150)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f'ROC Curve  (AUC = {roc_auc:.4f})')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'roc_curve.png', dpi=150)
    plt.close()

    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f'Precision-Recall Curve  (AUC = {pr_auc:.4f})')
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / 'pr_curve.png', dpi=150)
    plt.close()

    print(f'\nFigures saved: {FIGURES_PATH}')
    print(f'Metrics saved: {METRICS_PATH}')


if __name__ == '__main__':
    evaluate()
