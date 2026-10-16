"""Métricas comunes para todos los métodos."""

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def metricas(y_real, y_pred):
    return {
        "exactitud": float(accuracy_score(y_real, y_pred)),
        "f1_macro": float(f1_score(y_real, y_pred, average="macro")),
        "matriz_confusion": confusion_matrix(y_real, y_pred, labels=list(range(10))).tolist(),
    }


def latencia_ms(predecir_una, X, n=200, calentamiento=10):
    """Milisegundos por imagen al clasificar de una en una (como en la demo con webcam)."""
    for imagen in X[:calentamiento]:
        predecir_una(imagen[None])
    inicio = time.perf_counter()
    for imagen in X[:n]:
        predecir_una(imagen[None])
    return (time.perf_counter() - inicio) / n * 1000


def errores_mas_confiados(probabilidades, y_real, n=16):
    """Índices de los fallos en los que el modelo estaba más seguro de su predicción."""
    pred = probabilidades.argmax(axis=1)
    fallos = np.flatnonzero(pred != y_real)
    confianza = probabilidades[fallos, pred[fallos]]
    return fallos[np.argsort(-confianza)][:n]
