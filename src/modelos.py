"""Redes neuronales en PyTorch.

Todas reciben imágenes de 28x28 con valores en [0, 1] (forma N x 1 x 28 x 28) y
devuelven 10 puntuaciones (logits), una por dígito. Para convertirlas en
probabilidades se aplica softmax (lo hace `predecir_probabilidades`).
"""

import torch
from torch import nn

# Media y desviación típica de los píxeles de MNIST: se usan para normalizar la entrada.
MEDIA_MNIST, DESVIACION_MNIST = 0.1307, 0.3081


class Normalizar(nn.Module):
    """Primera capa de cada red: resta la media y divide entre la desviación de MNIST.

    Al hacerlo dentro del modelo, la web y la demo le pasan la imagen tal cual.
    """

    def forward(self, x):
        """Se ejecuta al pasar una imagen por la capa."""
        return (x - MEDIA_MNIST) / DESVIACION_MNIST


class MLP(nn.Module):
    """Perceptrón multicapa: 784 píxeles -> 256 neuronas ocultas -> 10 dígitos."""

    def __init__(self, ocultas=256):
        """Crea las capas (los pesos empiezan al azar)."""
        super().__init__()
        self.red = nn.Sequential(
            Normalizar(),
            nn.Flatten(),  # 28 x 28 -> vector de 784 valores
            nn.Linear(28 * 28, ocultas),
            nn.ReLU(),
            nn.Linear(ocultas, 10),
        )

    def forward(self, x):
        """Imágenes N x 1 x 28 x 28 -> 10 puntuaciones por imagen."""
        return self.red(x)


class LeNet(nn.Module):
    """Red convolucional basada en LeNet: tres convoluciones (6, 16 y 120 filtros) y dos capas densas.

    Con `mejorada=True` añade normalización por lotes (BatchNorm) tras cada convolución
    y dropout antes de la capa de salida. Es la versión que se entrena con aumento de
    datos y la que usan la web y la demo.
    """

    def __init__(self, mejorada=False, dropout=0.3):
        """Crea las capas (los pesos empiezan al azar)."""
        super().__init__()

        def bloque(entrada, salida, kernel):
            """Convolución (+ BatchNorm si es la versión mejorada) + ReLU."""
            capas = [nn.Conv2d(entrada, salida, kernel, padding=kernel // 2)]
            if mejorada:
                capas.append(nn.BatchNorm2d(salida))
            capas.append(nn.ReLU())
            return capas

        # Parte convolucional: detecta trazos y formas
        self.caracteristicas = nn.Sequential(
            Normalizar(),
            *bloque(1, 6, 5),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            *bloque(6, 16, 5),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            *bloque(16, 120, 3),
        )
        # Parte densa: decide el dígito a partir de esas formas
        self.clasificador = nn.Sequential(
            nn.Flatten(),  # 120 mapas de 7x7 -> vector de 5.880 valores
            nn.Linear(120 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout) if mejorada else nn.Identity(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        """Imágenes N x 1 x 28 x 28 -> 10 puntuaciones por imagen."""
        return self.clasificador(self.caracteristicas(x))


def crear_modelo(nombre):
    """Crea una red sin entrenar a partir de su nombre."""
    if nombre == "mlp":
        return MLP()
    if nombre == "lenet":
        return LeNet(mejorada=False)
    if nombre == "cnn_aumentada":
        return LeNet(mejorada=True)
    raise ValueError(f"Modelo desconocido: {nombre}")


def n_parametros(modelo):
    """Número de pesos que aprende la red."""
    return sum(p.numel() for p in modelo.parameters())


@torch.no_grad()  # al predecir no hacen falta gradientes
def predecir_probabilidades(modelo, X, lote=1000):
    """Probabilidad de cada dígito (forma N x 10) para un array de imágenes N x 28 x 28."""
    modelo.eval()  # modo evaluación: desactiva el dropout y fija la normalización por lotes
    salidas = []
    for i in range(0, len(X), lote):
        x = torch.as_tensor(X[i : i + lote], dtype=torch.float32)[:, None]  # añade el canal: N x 1 x 28 x 28
        salidas.append(torch.softmax(modelo(x), dim=1))
    return torch.cat(salidas).numpy()
