"""SSIM (índice de similitud estructural) de muchas parejas de imágenes a la vez.

El SSIM compara dos imágenes zona a zona (ventanas de 7x7 píxeles): mira si en cada
zona tienen el mismo brillo medio, el mismo contraste y la misma estructura. Vale 1
si las imágenes son idénticas y baja cuanto más se diferencian.

En cada zona, con μ = media, σ² = varianza y σab = covarianza:

    SSIM = (2·μa·μb + C1) · (2·σab + C2) / ((μa² + μb² + C1) · (σa² + σb² + C2))

y el resultado final es la media de todas las zonas.

Da lo mismo que `skimage.metrics.structural_similarity` con sus opciones por defecto
(lo comprueba tests/test_ssim.py), pero calcula miles de parejas a la vez con PyTorch:
unas 6 veces más rápido que ir pareja a pareja.
"""

import torch
import torch.nn.functional as F

VENTANA = 7  # lado de cada zona que se compara
C1 = (0.01 * 1.0) ** 2  # constantes que evitan dividir por cero (las de skimage,
C2 = (0.03 * 1.0) ** 2  # para imágenes con valores entre 0 y 1)
CORRECCION = VENTANA**2 / (VENTANA**2 - 1)  # varianza muestral (dividir entre n - 1), como skimage


def media_local(x):
    """Media de cada zona de 7x7 (sin relleno: una imagen de 28x28 da 22x22 zonas)."""
    return F.avg_pool2d(x, VENTANA, stride=1)


@torch.no_grad()  # no hace falta calcular gradientes: solo se mide la similitud
def ssim_por_pares(A, B, bloque_a=256, bloque_b=256):
    """Matriz con el SSIM de cada imagen de A frente a cada imagen de B.

    A y B son arrays de imágenes de 28x28 con valores en [0, 1]. El resultado tiene
    forma len(A) x len(B). Se calcula por bloques de 256 x 256 parejas para no
    llenar la memoria.
    """
    # 1. Media y varianza de cada zona, calculadas una sola vez por imagen.
    a = torch.as_tensor(A, dtype=torch.float32)[:, None]  # forma (nA, 1, 28, 28)
    b = torch.as_tensor(B, dtype=torch.float32)[:, None]  # forma (nB, 1, 28, 28)
    media_a, media_b = media_local(a), media_local(b)
    var_a = CORRECCION * (media_local(a * a) - media_a * media_a)
    var_b = CORRECCION * (media_local(b * b) - media_b * media_b)

    salida = torch.empty(len(a), len(b))
    for i in range(0, len(a), bloque_a):
        # Un bloque de A con una dimensión extra en la posición 1 ...
        a_i = a[i : i + bloque_a, None]  # (na, 1, 1, 28, 28)
        media_a_i = media_a[i : i + bloque_a, None]
        var_a_i = var_a[i : i + bloque_a, None]
        for j in range(0, len(b), bloque_b):
            # ... y un bloque de B con la dimensión extra en la posición 0: al operar
            # entre ellos, PyTorch combina cada imagen de A con cada imagen de B.
            b_j = b[None, j : j + bloque_b]  # (1, nb, 1, 28, 28)
            media_b_j = media_b[None, j : j + bloque_b]
            var_b_j = var_b[None, j : j + bloque_b]

            # 2. Covarianza de cada zona para cada pareja: media(a·b) − media(a)·media(b)
            producto = (a_i * b_j).flatten(0, 1)  # (na·nb, 1, 28, 28)
            media_ab = media_local(producto).view(a_i.shape[0], b_j.shape[1], 1, *media_a_i.shape[-2:])
            cov = CORRECCION * (media_ab - media_a_i * media_b_j)

            # 3. Fórmula del SSIM en cada zona y media de todas las zonas
            numerador = (2 * media_a_i * media_b_j + C1) * (2 * cov + C2)
            denominador = (media_a_i * media_a_i + media_b_j * media_b_j + C1) * (var_a_i + var_b_j + C2)
            salida[i : i + bloque_a, j : j + bloque_b] = (numerador / denominador).mean(dim=(-1, -2, -3))

    return salida.numpy()
