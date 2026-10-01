from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def confusion(truth, decided) -> dict[str, int]:
    """A matriz de confusao, com 1 como classe positiva."""
    truth = np.asarray(truth)
    decided = np.asarray(decided)

    return {
        "true_positives": int(((truth == 1) & (decided == 1)).sum()),
        "false_positives": int(((truth == 0) & (decided == 1)).sum()),
        "true_negatives": int(((truth == 0) & (decided == 0)).sum()),
        "false_negatives": int(((truth == 1) & (decided == 0)).sum()),
    }


def rates(counts: dict[str, int]) -> dict[str, float]:
    """F1, precisao, revocacao, acuracia e especificidade de uma matriz.

    Sem alerta nenhum, a precisao vale zero.
    """
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    tn = counts["true_negatives"]
    fn = counts["false_negatives"]
    alerts = tp + fp

    return {
        "f1": 2 * tp / (2 * tp + fp + fn),
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / (tp + fn),
        "accuracy": (tp + tn) / (tp + fp + tn + fn),
        "specificity": tn / (tn + fp),
    }


def roc_point(counts: dict[str, int]) -> tuple[float, float]:
    """A taxa de falsos positivos e a de verdadeiros positivos: o ponto no plano ROC.

    E o lugar de um classificador que so decide, sem escore para varrer limiar.
    """
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    tn = counts["true_negatives"]
    fn = counts["false_negatives"]

    return fp / (fp + tn), tp / (tp + fn)


def roc_auc(truth, score) -> float:
    """A area sob a curva ROC de um escore continuo."""
    return float(roc_auc_score(truth, score))


def negatives_per_positive(labels: pd.Series) -> float:
    """Quantas negativas ha para cada positiva: o peso que equilibra a classe rara."""
    positives = int(labels.sum())
    negatives = len(labels) - positives

    return negatives / positives
