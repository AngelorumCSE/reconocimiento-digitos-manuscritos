"""Métricas comunes para comparar todos los métodos de la misma forma."""

import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def metricas(y_real, y_pred):
    """Exactitud, F1 macro y matriz de confusión.

    - exactitud: fracción de imágenes bien clasificadas;
    - F1 macro: media del F1 de cada dígito (penaliza si algún dígito va mucho peor);
    - matriz de confusión: cuántas imágenes de cada dígito real se predicen como cada dígito.
    """
    return {
        "exactitud": float(accuracy_score(y_real, y_pred)),
        "f1_macro": float(f1_score(y_real, y_pred, average="macro")),
        "matriz_confusion": confusion_matrix(y_real, y_pred, labels=list(range(10))).tolist(),
    }


def latencia_ms(predecir_una, X, n=200, calentamiento=10):
    """Milisegundos por imagen al clasificar de una en una, como en la demo con webcam.

    Las primeras llamadas suelen ser más lentas (se reserva memoria), así que se
    hacen unas cuantas de calentamiento antes de medir.
    """
    for imagen in X[:calentamiento]:
        predecir_una(imagen[None])
    inicio = time.perf_counter()
    for imagen in X[:n]:
        predecir_una(imagen[None])
    return (time.perf_counter() - inicio) / n * 1000


def errores_mas_confiados(probabilidades, y_real, n=16):
    """Índices de los fallos en los que la red estaba más segura de su respuesta."""
    pred = probabilidades.argmax(axis=1)
    fallos = np.flatnonzero(pred != y_real)
    confianza = probabilidades[fallos, pred[fallos]]
    return fallos[np.argsort(-confianza)][:n]  # de más a menos confianza
