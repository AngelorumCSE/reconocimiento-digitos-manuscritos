"""De una foto, un fotograma de la webcam o un dibujo a una imagen de 28x28 al estilo MNIST.

MNIST tiene el dígito en blanco sobre fondo negro, encajado en una caja de 20x20
píxeles (sin deformarlo) y centrado por su centro de masas en una imagen de 28x28.
Para que la red funcione con imágenes reales hay que reproducir ese proceso:

1. escala de grises y suavizado;
2. umbral adaptativo (aguanta sombras y luz desigual) y eliminación de ruido; si la hoja
   se ve entera, solo se buscan trazos dentro de ella;
3. selección de los trazos del dígito (se descartan manchas y lo que toca el borde);
4. ajuste del grosor del trazo (un bolígrafo fino queda demasiado fino al reducirlo);
5. reducción a 20x20 manteniendo la proporción y centrado por centro de masas en 28x28.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

LADO_SALIDA = 28
LADO_DIGITO = 20
GROSOR_OBJETIVO = 0.13  # grosor de trazo típico en MNIST respecto al lado de la caja de 20 px
AREA_MINIMA = 0.002  # fracción del área de la imagen por debajo de la cual una mancha es ruido
AREA_MINIMA_HOJA = 0.10  # una hoja que se ve entera ocupa al menos esta fracción de la imagen
LADO_MAXIMO = 1200  # píxeles: las fotos más grandes se reducen antes de procesarlas


@dataclass
class Resultado:
    imagen_28: np.ndarray | None  # float32 28x28 en [0, 1], o None si no hay dígito
    binaria: np.ndarray  # máscara del umbral (uint8 0/255)
    caja: tuple | None = None  # (x, y, ancho, alto) del dígito en la imagen de entrada
    pasos: dict = field(default_factory=dict)  # imágenes intermedias para explicar el proceso


def a_gris(imagen):
    if imagen.ndim == 3:
        codigo = cv2.COLOR_BGRA2GRAY if imagen.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        imagen = cv2.cvtColor(imagen, codigo)
    if imagen.dtype != np.uint8:
        imagen = np.clip(imagen * (255 if imagen.max() <= 1 else 1), 0, 255).astype(np.uint8)
    return imagen


def interior_de_la_hoja(suavizada):
    """Interior de la hoja de papel si se ve entera dentro de la imagen; None en otro caso.

    Con el umbral adaptativo, el salto de luz entre el papel y la mesa deja un marco alrededor
    de la hoja que la red confundiría con un 0. Si la hoja llena la imagen o se sale de ella,
    ese borde toca el borde de la imagen y ya se descarta al seleccionar el dígito.
    """
    alto, ancho = suavizada.shape
    _, claro = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(claro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contorno) < AREA_MINIMA_HOJA * alto * ancho:
            break
        x, y, w, h = cv2.boundingRect(contorno)
        if x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1:
            continue
        interior = np.zeros_like(claro)
        cv2.drawContours(interior, [contorno], -1, 255, thickness=cv2.FILLED)
        tinta = cv2.countNonZero(cv2.bitwise_and(interior, cv2.bitwise_not(claro)))
        if tinta < AREA_MINIMA * alto * ancho:
            continue  # una zona clara sin nada escrito (una lámpara, una pared) no es la hoja
        k = max(5, int(0.04 * min(w, h))) | 1
        return cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return None


def binarizar(gris, fondo="auto"):
    """Máscara con el trazo en blanco.

    fondo: "claro" (tinta oscura sobre papel, como con la webcam), "oscuro" (trazo claro sobre
    fondo oscuro, como en MNIST) o "auto" (claro si la mayor parte de la imagen es clara).
    """
    lado = min(gris.shape)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)
    fondo_claro = fondo == "claro" or (fondo == "auto" and np.median(suavizada) > 127)
    if fondo_claro:
        bloque = max(15, (lado // 8) | 1)  # tamaño impar proporcional a la imagen
        mascara = cv2.adaptiveThreshold(
            suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bloque, 10
        )
        hoja = interior_de_la_hoja(suavizada)
        if hoja is not None:
            mascara = cv2.bitwise_and(mascara, hoja)
    else:
        _, mascara = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)


def seleccionar_digito(mascara):
    """Se queda con los trazos relevantes: la mancha mayor y las cercanas de tamaño suficiente."""
    alto, ancho = mascara.shape
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:
        return None, None

    area_min = AREA_MINIMA * alto * ancho
    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        toca_borde = x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1
        if area >= area_min and not toca_borde:
            candidatos.append(i)
    if not candidatos:
        return None, None

    principal = max(candidatos, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    px, py, pw, ph, parea = stats[principal]
    margen = 0.5 * max(pw, ph)
    elegidos = []
    for i in candidatos:
        x, y, w, h, area = stats[i]
        cerca = (x < px + pw + margen and x + w > px - margen and y < py + ph + margen and y + h > py - margen)
        if i == principal or (cerca and area >= 0.08 * parea):
            elegidos.append(i)

    seleccion = np.isin(etiquetas, elegidos).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(seleccion)
    return seleccion, (x, y, w, h)


def ajustar_grosor(recorte):
    """Engrosa trazos finos para que, al reducir a 20x20, se parezcan a los de MNIST."""
    area = cv2.countNonZero(recorte)
    contornos, _ = cv2.findContours(recorte, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimetro = sum(cv2.arcLength(c, True) for c in contornos)
    if area == 0 or perimetro == 0:
        return recorte
    grosor = 2 * area / perimetro  # aproximación: área ≈ longitud x grosor
    objetivo = GROSOR_OBJETIVO * max(recorte.shape)
    if grosor < 0.75 * objetivo:
        k = int(round(objetivo - grosor)) | 1
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        recorte = cv2.dilate(recorte, nucleo)
    return recorte


def centrar_como_mnist(recorte):
    """Encaja el dígito en 20x20 sin deformarlo y lo centra por su centro de masas en 28x28."""
    alto, ancho = recorte.shape
    factor = LADO_DIGITO / max(alto, ancho)
    nuevo = (max(1, int(round(ancho * factor))), max(1, int(round(alto * factor))))
    reducido = cv2.resize(recorte, nuevo, interpolation=cv2.INTER_AREA)

    lienzo = np.zeros((LADO_SALIDA, LADO_SALIDA), np.float32)
    y0 = (LADO_SALIDA - reducido.shape[0]) // 2
    x0 = (LADO_SALIDA - reducido.shape[1]) // 2
    lienzo[y0 : y0 + reducido.shape[0], x0 : x0 + reducido.shape[1]] = reducido / 255.0

    momentos = cv2.moments(lienzo)
    if momentos["m00"] > 0:
        cx, cy = momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]
        desplazamiento = np.float32([[1, 0, LADO_SALIDA / 2 - cx], [0, 1, LADO_SALIDA / 2 - cy]])
        lienzo = cv2.warpAffine(lienzo, desplazamiento, (LADO_SALIDA, LADO_SALIDA))
    return np.clip(lienzo, 0, 1)


def recuadro_central(imagen, fraccion=0.6):
    """Recuadro cuadrado y centrado (x, y, lado) donde la demo y la web buscan el dígito."""
    alto, ancho = imagen.shape[:2]
    lado = int(min(alto, ancho) * fraccion)
    return (ancho - lado) // 2, (alto - lado) // 2, lado


def preprocesar(imagen, fondo="auto"):
    """Imagen BGR/gris (uint8) -> Resultado con la imagen 28x28 lista para la red.

    Con la webcam o la cámara, usa fondo="claro": siempre es tinta oscura sobre papel.
    Las fotos grandes (las de un móvil) se reducen antes, para ir más rápido; la caja del
    dígito se devuelve siempre en coordenadas de la imagen original.
    """
    escala = 1.0
    alto, ancho = imagen.shape[:2]
    if max(alto, ancho) > LADO_MAXIMO:
        escala = LADO_MAXIMO / max(alto, ancho)
        imagen = cv2.resize(imagen, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)

    gris = a_gris(imagen)
    mascara = binarizar(gris, fondo)
    seleccion, caja = seleccionar_digito(mascara)
    if seleccion is None:
        return Resultado(None, mascara)

    x, y, w, h = caja
    lado = max(w, h)
    margen = int(0.1 * lado) + 2
    recorte = seleccion[max(0, y - margen) : y + h + margen, max(0, x - margen) : x + w + margen]
    recorte = ajustar_grosor(recorte)
    # tras engrosar, se recorta de nuevo al contenido para que el dígito ocupe la caja de 20x20
    rx, ry, rw, rh = cv2.boundingRect(recorte)
    recorte = recorte[ry : ry + rh, rx : rx + rw]
    imagen_28 = centrar_como_mnist(recorte)

    pasos = {"gris": gris, "binaria": mascara, "seleccion": seleccion, "recorte": recorte, "entrada_red": imagen_28}
    if escala != 1.0:
        caja = tuple(int(round(v / escala)) for v in caja)
    return Resultado(imagen_28, mascara, caja, pasos)
