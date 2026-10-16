"""Entrenamiento de las redes con validación, parada temprana y aumento de datos opcional."""

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .datos import SEMILLA
from .modelos import predecir_probabilidades

# Aumento de datos: giros, desplazamientos, cambios de escala y trazos más gruesos,
# para que la red aguante dígitos escritos a mano y fotografiados con la webcam.
AUM_GIRO = 20.0  # grados
AUM_DESPLAZAMIENTO = 3.0  # píxeles
AUM_ESCALA = (0.85, 1.15)
AUM_PROB_ENGROSAR = 0.2


def aumentar_lote(x, generador):
    """Transformación afín aleatoria distinta para cada imagen del lote (en la propia CPU/GPU)."""
    n = x.shape[0]
    angulo = (torch.rand(n, generator=generador) * 2 - 1) * math.radians(AUM_GIRO)
    escala = torch.empty(n).uniform_(*AUM_ESCALA, generator=generador)
    desplazamiento = (torch.rand(n, 2, generator=generador) * 2 - 1) * AUM_DESPLAZAMIENTO * 2 / 28

    cos, sen = torch.cos(angulo) / escala, torch.sin(angulo) / escala
    theta = torch.stack(
        [
            torch.stack([cos, -sen, desplazamiento[:, 0]], dim=1),
            torch.stack([sen, cos, desplazamiento[:, 1]], dim=1),
        ],
        dim=1,
    )
    rejilla = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, rejilla, align_corners=False, padding_mode="zeros")

    engrosar = torch.rand(n, generator=generador) < AUM_PROB_ENGROSAR
    if engrosar.any():
        x[engrosar] = F.max_pool2d(x[engrosar], 3, stride=1, padding=1)
    return x


def exactitud(modelo, X, y):
    return float((predecir_probabilidades(modelo, X).argmax(axis=1) == y).mean())


def entrenar(
    modelo,
    X,
    y,
    X_val,
    y_val,
    epocas=15,
    pasos_minimos=0,
    lote=128,
    lr=1e-3,
    aumentar=False,
    paciencia=3,
    evaluar_cada=None,
    semilla=SEMILLA,
    verbose=True,
):
    """Entrena con Adam y tasa de aprendizaje coseno; se queda con los pesos de mejor validación.

    `pasos_minimos` garantiza suficientes actualizaciones cuando hay pocas muestras
    (curva de aprendizaje). La parada temprana usa solo el conjunto de validación.
    """
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)

    X_t = torch.as_tensor(X, dtype=torch.float32)[:, None]
    y_t = torch.as_tensor(y, dtype=torch.long)
    n = len(y_t)
    pasos_por_epoca = math.ceil(n / lote)
    total_pasos = max(epocas * pasos_por_epoca, pasos_minimos)
    evaluar_cada = evaluar_cada or pasos_por_epoca

    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=total_pasos)
    perdida_fn = nn.CrossEntropyLoss()

    historial = {"paso": [], "perdida_train": [], "exactitud_val": []}
    mejor = {"exactitud_val": -1.0, "paso": 0, "pesos": None}
    sin_mejora = 0
    paso = 0
    perdidas = []
    inicio = time.perf_counter()

    while paso < total_pasos:
        orden = torch.randperm(n, generator=generador)
        for i in range(0, n, lote):
            modelo.train()
            indices = orden[i : i + lote]
            xb, yb = X_t[indices], y_t[indices]
            if aumentar:
                xb = aumentar_lote(xb, generador)

            optimizador.zero_grad()
            perdida = perdida_fn(modelo(xb), yb)
            perdida.backward()
            optimizador.step()
            planificador.step()
            perdidas.append(perdida.item())
            paso += 1

            if paso % evaluar_cada == 0 or paso == total_pasos:
                acc_val = exactitud(modelo, X_val, y_val)
                historial["paso"].append(paso)
                historial["perdida_train"].append(float(np.mean(perdidas)))
                historial["exactitud_val"].append(acc_val)
                perdidas = []
                if verbose:
                    print(f"  paso {paso:5d}/{total_pasos}  val={acc_val:.4f}", flush=True)
                if acc_val > mejor["exactitud_val"]:
                    mejor = {"exactitud_val": acc_val, "paso": paso, "pesos": copy.deepcopy(modelo.state_dict())}
                    sin_mejora = 0
                else:
                    sin_mejora += 1
                if sin_mejora >= paciencia:
                    total_pasos = paso  # parada temprana
            if paso >= total_pasos:
                break

    modelo.load_state_dict(mejor["pesos"])
    modelo.eval()
    return {
        "historial": historial,
        "mejor_exactitud_val": mejor["exactitud_val"],
        "mejor_paso": mejor["paso"],
        "pasos": paso,
        "pasos_por_epoca": pasos_por_epoca,
        "segundos": time.perf_counter() - inicio,
    }
