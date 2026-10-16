"""Entrenamiento de las redes: Adam, validación, parada temprana y aumento de datos.

Cómo se entrena una red, paso a paso:

1. Se baraja el entrenamiento y se recorre en lotes de 128 imágenes.
2. Con cada lote: la red predice, se calcula la pérdida (entropía cruzada), se
   calculan los gradientes y el optimizador Adam ajusta los pesos.
3. Tras cada epoch (una pasada por todo el entrenamiento) se mide la exactitud
   en validación y se guardan los pesos si es la mejor hasta ahora.
4. Si la validación no mejora en varios epochs seguidos («paciencia»), se para antes
   (parada temprana). Al final, la red se queda con los pesos de mejor validación.
"""

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
AUM_PROB_ENGROSAR = 0.2  # el 20 % de las imágenes recibe un trazo más grueso


def aumentar_lote(x, generador):
    """Aplica a cada imagen del lote una transformación aleatoria distinta.

    x tiene forma (n, 1, 28, 28). Se usa una transformación afín (giro + escala +
    desplazamiento) para todo el lote a la vez, que es mucho más rápido que ir
    imagen a imagen.
    """
    n = x.shape[0]
    # Un giro, una escala y un desplazamiento al azar para cada imagen
    angulo = (torch.rand(n, generator=generador) * 2 - 1) * math.radians(AUM_GIRO)
    escala = torch.empty(n).uniform_(*AUM_ESCALA, generator=generador)
    # affine_grid mide el desplazamiento en unidades de -1 a 1 (la imagen mide 2): por eso * 2 / 28
    desplazamiento = (torch.rand(n, 2, generator=generador) * 2 - 1) * AUM_DESPLAZAMIENTO * 2 / 28

    # Matriz de la transformación de cada imagen: [[cos, -sen, dx], [sen, cos, dy]]
    cos, sen = torch.cos(angulo) / escala, torch.sin(angulo) / escala
    theta = torch.stack(
        [
            torch.stack([cos, -sen, desplazamiento[:, 0]], dim=1),
            torch.stack([sen, cos, desplazamiento[:, 1]], dim=1),
        ],
        dim=1,
    )
    rejilla = F.affine_grid(theta, x.shape, align_corners=False)  # dónde va a parar cada píxel
    x = F.grid_sample(x, rejilla, align_corners=False, padding_mode="zeros")  # imagen transformada

    # Trazo más grueso en algunas imágenes: cada píxel toma el máximo de sus vecinos (dilatación)
    engrosar = torch.rand(n, generator=generador) < AUM_PROB_ENGROSAR
    if engrosar.any():
        x[engrosar] = F.max_pool2d(x[engrosar], 3, stride=1, padding=1)
    return x


def exactitud(modelo, X, y):
    """Fracción de imágenes que la red clasifica bien."""
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
    """Entrena la red y la deja con los pesos de mejor exactitud en validación.

    Parámetros:
    - epocas: número máximo de pasadas por el entrenamiento.
    - pasos_minimos: actualizaciones mínimas de los pesos. Solo se usa en la curva de
      aprendizaje, cuando hay tan pocas imágenes que 15 epochs serían muy pocos pasos.
    - lote: imágenes por paso; lr: tasa de aprendizaje inicial de Adam.
    - aumentar: si es True, cada lote pasa por `aumentar_lote` antes de entrar a la red.
    - paciencia: evaluaciones seguidas sin mejorar antes de parar (parada temprana).
    - evaluar_cada: cada cuántos pasos se mide la validación (por defecto, cada epoch).

    Devuelve un diccionario con el historial de la validación, la mejor exactitud,
    el paso en el que se consiguió y los segundos que ha tardado.
    """
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)  # mismo orden y mismo aumento en cada ejecución

    X_t = torch.as_tensor(X, dtype=torch.float32)[:, None]  # forma N x 1 x 28 x 28
    y_t = torch.as_tensor(y, dtype=torch.long)
    n = len(y_t)
    pasos_por_epoca = math.ceil(n / lote)
    total_pasos = max(epocas * pasos_por_epoca, pasos_minimos)
    evaluar_cada = evaluar_cada or pasos_por_epoca

    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    # La tasa de aprendizaje baja poco a poco siguiendo medio coseno, hasta casi 0 al final
    planificador = torch.optim.lr_scheduler.CosineAnnealingLR(optimizador, T_max=total_pasos)
    perdida_fn = nn.CrossEntropyLoss()  # aplica la softmax y mide lo lejos que está de la etiqueta

    historial = {"paso": [], "perdida_train": [], "exactitud_val": []}
    mejor = {"exactitud_val": -1.0, "paso": 0, "pesos": None}
    sin_mejora = 0
    paso = 0
    perdidas = []
    inicio = time.perf_counter()

    while paso < total_pasos:
        orden = torch.randperm(n, generator=generador)  # se baraja el entrenamiento en cada epoch
        for i in range(0, n, lote):
            # --- un paso de entrenamiento con un lote ---
            modelo.train()
            indices = orden[i : i + lote]
            xb, yb = X_t[indices], y_t[indices]
            if aumentar:
                xb = aumentar_lote(xb, generador)

            optimizador.zero_grad()  # borra los gradientes del paso anterior
            perdida = perdida_fn(modelo(xb), yb)  # predicción y error
            perdida.backward()  # gradientes: cómo cambiar cada peso para reducir el error
            optimizador.step()  # Adam ajusta los pesos
            planificador.step()  # y se baja un poco la tasa de aprendizaje
            perdidas.append(perdida.item())
            paso += 1

            # --- de vez en cuando, se mide la validación ---
            if paso % evaluar_cada == 0 or paso == total_pasos:
                acc_val = exactitud(modelo, X_val, y_val)
                historial["paso"].append(paso)
                historial["perdida_train"].append(float(np.mean(perdidas)))
                historial["exactitud_val"].append(acc_val)
                perdidas = []
                if verbose:
                    print(f"  paso {paso:5d}/{total_pasos}  val={acc_val:.4f}", flush=True)
                if acc_val > mejor["exactitud_val"]:
                    # la mejor hasta ahora: se guarda una copia de los pesos
                    mejor = {"exactitud_val": acc_val, "paso": paso, "pesos": copy.deepcopy(modelo.state_dict())}
                    sin_mejora = 0
                else:
                    sin_mejora += 1
                if sin_mejora >= paciencia:
                    total_pasos = paso  # parada temprana: se termina aquí
            if paso >= total_pasos:
                break

    modelo.load_state_dict(mejor["pesos"])  # se recuperan los pesos de mejor validación
    modelo.eval()
    return {
        "historial": historial,
        "mejor_exactitud_val": mejor["exactitud_val"],
        "mejor_paso": mejor["paso"],
        "pasos": paso,
        "pasos_por_epoca": pasos_por_epoca,
        "segundos": time.perf_counter() - inicio,
    }
