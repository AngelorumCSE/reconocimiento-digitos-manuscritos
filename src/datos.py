"""Carga de MNIST y partición de los datos.

Todos los métodos del proyecto usan exactamente la misma partición:

- entrenamiento: 55.000 imágenes del conjunto oficial de entrenamiento;
- validación: 5.000 imágenes del conjunto oficial de entrenamiento, que solo sirven
  para elegir hiperparámetros y el mejor epoch;
- test: las 10.000 imágenes del conjunto oficial de test, que no se usan nunca para
  entrenar ni para tomar decisiones.

También genera el «test perturbado»: el mismo test con cada dígito girado, desplazado
y escalado al azar, para medir lo robusto que es cada método.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

RUTA_DATOS = Path(__file__).resolve().parent.parent / "data"  # MNIST se descarga aquí (unos 60 MB)
SEMILLA = 42  # semilla fija: la partición y los resultados salen siempre iguales
N_VALIDACION = 5000

# Perturbaciones del test de robustez, parecidas a las de un dígito escrito a mano
# y fotografiado con la webcam.
GIRO_MAX = 25.0  # grados
DESPLAZAMIENTO_MAX = 3  # píxeles
ESCALA_MIN, ESCALA_MAX = 0.8, 1.2


def cargar_mnist(ruta=RUTA_DATOS):
    """Descarga MNIST (solo la primera vez) y devuelve la partición en un diccionario.

    Las imágenes (X) son arrays float32 de forma N x 28 x 28 con valores en [0, 1];
    las etiquetas (y) son el dígito de cada imagen (0 a 9).
    """
    entrenamiento = datasets.MNIST(ruta, train=True, download=True)
    test = datasets.MNIST(ruta, train=False, download=True)

    # Los píxeles vienen de 0 a 255: se dividen entre 255 para tenerlos entre 0 y 1.
    X = entrenamiento.data.numpy().astype(np.float32) / 255.0
    y = entrenamiento.targets.numpy().astype(np.int64)
    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    # Se separan 5.000 imágenes para validación. «stratify» mantiene la misma
    # proporción de cada dígito en las dos partes.
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
    """Toma n imágenes al azar manteniendo la proporción de cada dígito."""
    if n >= len(y):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=semilla)
    return X_sub, y_sub


def perturbar(X, semilla=SEMILLA):
    """Gira, desplaza y escala cada imagen al azar (siempre igual gracias a la semilla)."""
    rng = np.random.default_rng(semilla)
    salida = np.empty_like(X)
    for i, imagen in enumerate(X):
        # Valores al azar para esta imagen
        angulo = float(rng.uniform(-GIRO_MAX, GIRO_MAX))
        dx, dy = (int(v) for v in rng.integers(-DESPLAZAMIENTO_MAX, DESPLAZAMIENTO_MAX + 1, size=2))
        escala = float(rng.uniform(ESCALA_MIN, ESCALA_MAX))

        # Transformación afín (giro + desplazamiento + escala) con torchvision
        tensor = torch.from_numpy(imagen)[None]  # torchvision espera la forma 1 x 28 x 28
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
