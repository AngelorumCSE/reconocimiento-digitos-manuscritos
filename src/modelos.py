"""Redes neuronales en PyTorch.

Todas reciben imágenes de 28x28 con valores en [0, 1] (forma N x 1 x 28 x 28) y
devuelven logits: la softmax la aplica la función de pérdida durante el
entrenamiento y `torch.softmax` al predecir.
"""

import torch
from torch import nn

MEDIA_MNIST, DESVIACION_MNIST = 0.1307, 0.3081


class Normalizar(nn.Module):
    """Normaliza dentro del modelo para que la demo y ONNX reciban la imagen tal cual."""

    def forward(self, x):
        return (x - MEDIA_MNIST) / DESVIACION_MNIST


class MLP(nn.Module):
    """Perceptrón multicapa: 784 -> 256 -> 10."""

    def __init__(self, ocultas=256):
        super().__init__()
        self.red = nn.Sequential(
            Normalizar(),
            nn.Flatten(),
            nn.Linear(28 * 28, ocultas),
            nn.ReLU(),
            nn.Linear(ocultas, 10),
        )

    def forward(self, x):
        return self.red(x)


class LeNet(nn.Module):
    """Variante de LeNet de la práctica: tres convoluciones (6, 16 y 120 filtros) y dos capas densas.

    Con `mejorada=True` añade normalización por lotes tras cada convolución y dropout
    antes de la capa de salida. Es la versión que se entrena con aumento de datos.
    """

    def __init__(self, mejorada=False, dropout=0.3):
        super().__init__()

        def bloque(entrada, salida, kernel):
            capas = [nn.Conv2d(entrada, salida, kernel, padding=kernel // 2)]
            if mejorada:
                capas.append(nn.BatchNorm2d(salida))
            capas.append(nn.ReLU())
            return capas

        self.caracteristicas = nn.Sequential(
            Normalizar(),
            *bloque(1, 6, 5),
            nn.MaxPool2d(2),  # 28 -> 14
            *bloque(6, 16, 5),
            nn.MaxPool2d(2),  # 14 -> 7
            *bloque(16, 120, 3),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout) if mejorada else nn.Identity(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.clasificador(self.caracteristicas(x))


def crear_modelo(nombre):
    if nombre == "mlp":
        return MLP()
    if nombre == "lenet":
        return LeNet(mejorada=False)
    if nombre == "cnn_aumentada":
        return LeNet(mejorada=True)
    raise ValueError(f"Modelo desconocido: {nombre}")


def n_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


@torch.no_grad()
def predecir_probabilidades(modelo, X, lote=1000):
    """Probabilidades (N x 10) para un array de imágenes N x 28 x 28 en [0, 1]."""
    modelo.eval()
    salidas = []
    for i in range(0, len(X), lote):
        x = torch.as_tensor(X[i : i + lote], dtype=torch.float32)[:, None]
        salidas.append(torch.softmax(modelo(x), dim=1))
    return torch.cat(salidas).numpy()
