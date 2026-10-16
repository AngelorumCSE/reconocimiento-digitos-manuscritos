"""Carga de MNIST, partición en entrenamiento/validación/test y conjunto de test perturbado.

Todas las técnicas del proyecto usan exactamente la misma partición:

- entrenamiento: 55.000 imágenes del conjunto oficial de entrenamiento;
- validación: 5.000 imágenes (estratificadas) del conjunto oficial de entrenamiento,
  usadas solo para elegir hiperparámetros y el mejor epoch;
- test: las 10.000 imágenes del conjunto oficial de test, que no se usan nunca
  para entrenar ni para tomar decisiones.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

RUTA_DATOS = Path(__file__).resolve().parent.parent / "data"
SEMILLA = 42
N_VALIDACION = 5000

# Perturbaciones del test de robustez: giro, desplazamiento y cambio de escala
# parecidos a los de un dígito escrito a mano y fotografiado con la webcam.
GIRO_MAX = 25.0  # grados
DESPLAZAMIENTO_MAX = 3  # píxeles
ESCALA_MIN, ESCALA_MAX = 0.8, 1.2


def cargar_mnist(ruta=RUTA_DATOS):
    """Devuelve un diccionario con X (float32 en [0, 1], forma N x 28 x 28) e y (int64)."""
    entrenamiento = datasets.MNIST(ruta, train=True, download=True)
    test = datasets.MNIST(ruta, train=False, download=True)

    X = entrenamiento.data.numpy().astype(np.float32) / 255.0
    y = entrenamiento.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=N_VALIDACION, stratify=y, random_state=SEMILLA
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def subconjunto_estratificado(X, y, n, semilla=SEMILLA):
    """Toma n muestras manteniendo la proporción de cada dígito."""
    if n >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=semilla)
    return X_sub, y_sub


def perturbar(X, semilla=SEMILLA):
    """Aplica a cada imagen un giro, un desplazamiento y un escalado aleatorios (reproducibles)."""
    rng = np.random.default_rng(semilla)
    salida = np.empty_like(X)
    for i, imagen in enumerate(X):
        angulo = float(rng.uniform(-GIRO_MAX, GIRO_MAX))
        dx, dy = (int(v) for v in rng.integers(-DESPLAZAMIENTO_MAX, DESPLAZAMIENTO_MAX + 1, size=2))
        escala = float(rng.uniform(ESCALA_MIN, ESCALA_MAX))
        tensor = torch.from_numpy(imagen)[None]
        transformada = TF.affine(
            tensor,
            angle=angulo,
            translate=[dx, dy],
            scale=escala,
            shear=[0.0],
            interpolation=InterpolationMode.BILINEAR,
        )
        salida[i] = transformada[0].numpy()
    return salida
