"""De una foto (o un fotograma de la webcam) a una imagen de 28x28 al estilo MNIST.

La red solo ha visto imágenes de MNIST: dígito blanco sobre fondo negro, encajado en
una caja de 20x20 píxeles (sin deformarlo) y centrado por su centro de masas en una
imagen de 28x28. Para que funcione con fotos reales hay que preparar la foto igual:

1. Escala de grises y suavizado (quita ruido).
2. Umbral adaptativo: separa la tinta del papel aunque haya sombras o luz desigual.
   Si la hoja se ve entera, solo se buscan trazos dentro de ella.
3. Selección de los trazos del dígito: se descartan manchas pequeñas y lo que toca
   el borde de la imagen.
4. Ajuste del grosor: un bolígrafo fino quedaría demasiado fino al reducir la imagen.
5. Reducción a 20x20 manteniendo la proporción y centrado en 28x28.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

LADO_SALIDA = 28  # tamaño de las imágenes de MNIST
LADO_DIGITO = 20  # en MNIST, el dígito ocupa una caja de 20x20 dentro de la imagen
GROSOR_OBJETIVO = 0.13  # grosor típico del trazo en MNIST, como fracción del tamaño del dígito
AREA_MINIMA = 0.002  # una mancha con menos de este área (fracción de la imagen) se toma por ruido
AREA_MINIMA_HOJA = 0.10  # una hoja que se ve entera ocupa al menos el 10 % de la imagen
LADO_MAXIMO = 1200  # las fotos más grandes (las del móvil) se reducen antes, para ir más rápido


@dataclass
class Resultado:
    """Lo que devuelve `preprocesar`."""

    imagen_28: np.ndarray | None  # imagen lista para la red (float32 28x28 en [0, 1]), o None si no hay dígito
    binaria: np.ndarray  # máscara del umbral: trazos en blanco (255) y fondo en negro (0)
    caja: tuple | None = None  # (x, y, ancho, alto) del dígito en la imagen de entrada
    pasos: dict = field(default_factory=dict)  # imágenes intermedias, para enseñar el proceso


def a_gris(imagen):
    """Convierte la imagen a escala de grises con valores enteros de 0 a 255."""
    if imagen.ndim == 3:  # imagen en color (OpenCV guarda los colores en orden BGR)
        codigo = cv2.COLOR_BGRA2GRAY if imagen.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        imagen = cv2.cvtColor(imagen, codigo)
    if imagen.dtype != np.uint8:  # imagen con valores de 0 a 1: se pasa a 0-255
        imagen = np.clip(imagen * (255 if imagen.max() <= 1 else 1), 0, 255).astype(np.uint8)
    return imagen


def interior_de_la_hoja(suavizada):
    """Máscara del interior de la hoja de papel si se ve entera; None si no.

    Con la webcam, la hoja suele verse entera sobre la mesa. El salto de luz entre el
    papel y la mesa dejaría un marco alrededor de la hoja que la red tomaría por un 0,
    así que en ese caso solo se buscan trazos dentro de la hoja.
    """
    alto, ancho = suavizada.shape
    # Otsu elige solo el umbral que separa lo claro (papel) de lo oscuro (mesa)
    _, claro = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(claro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True):  # de mayor a menor
        if cv2.contourArea(contorno) < AREA_MINIMA_HOJA * alto * ancho:
            break  # el resto es todavía más pequeño: no hay hoja
        x, y, w, h = cv2.boundingRect(contorno)
        if x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1:
            continue  # toca el borde: la hoja no se ve entera (ese caso ya lo resuelve seleccionar_digito)
        interior = np.zeros_like(claro)
        cv2.drawContours(interior, [contorno], -1, 255, thickness=cv2.FILLED)
        tinta = cv2.countNonZero(cv2.bitwise_and(interior, cv2.bitwise_not(claro)))
        if tinta < AREA_MINIMA * alto * ancho:
            continue  # una zona clara sin nada escrito (una lámpara, una pared) no es la hoja
        # Se encoge un poco la máscara para quedarse lejos del borde del papel
        k = max(5, int(0.04 * min(w, h))) | 1
        return cv2.erode(interior, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return None


def binarizar(gris, fondo="auto"):
    """Máscara con los trazos en blanco y el fondo en negro.

    fondo: "claro" (tinta oscura sobre papel, como con la webcam), "oscuro" (trazo
    claro sobre fondo oscuro, como en MNIST) o "auto" (claro si la mayor parte de la
    imagen es clara).
    """
    lado = min(gris.shape)
    suavizada = cv2.GaussianBlur(gris, (5, 5), 0)  # el suavizado quita el ruido de la cámara
    fondo_claro = fondo == "claro" or (fondo == "auto" and np.median(suavizada) > 127)
    if fondo_claro:
        # Umbral adaptativo: cada píxel se compara con la media de su vecindario, así que
        # aguanta sombras y luz desigual. INV deja la tinta (oscura) en blanco.
        bloque = max(15, (lado // 8) | 1)  # tamaño del vecindario: impar y proporcional a la imagen
        mascara = cv2.adaptiveThreshold(
            suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bloque, 10
        )
        hoja = interior_de_la_hoja(suavizada)
        if hoja is not None:
            mascara = cv2.bitwise_and(mascara, hoja)  # solo los trazos dentro de la hoja
    else:
        # Fondo oscuro: basta con un umbral global (Otsu)
        _, mascara = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # «Apertura» morfológica: borra los puntitos sueltos de ruido
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)


def seleccionar_digito(mascara):
    """Se queda con los trazos del dígito: la mancha mayor y las cercanas de tamaño suficiente.

    Devuelve (máscara solo con el dígito, caja (x, y, ancho, alto)), o (None, None).
    """
    alto, ancho = mascara.shape
    # Separa la máscara en manchas (componentes conexas) y mide cada una
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    if n <= 1:  # solo hay fondo
        return None, None

    # Candidatas: manchas suficientemente grandes que no tocan el borde de la imagen
    area_min = AREA_MINIMA * alto * ancho
    candidatos = []
    for i in range(1, n):  # la 0 es el fondo
        x, y, w, h, area = stats[i]
        toca_borde = x <= 1 or y <= 1 or x + w >= ancho - 1 or y + h >= alto - 1
        if area >= area_min and not toca_borde:
            candidatos.append(i)
    if not candidatos:
        return None, None

    # La mancha mayor es el dígito; se añaden las cercanas con al menos el 8 % de su área
    # (por ejemplo, el palito de un 4 o de un 7 que ha quedado separado).
    principal = max(candidatos, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    px, py, pw, ph, parea = stats[principal]
    margen = 0.5 * max(pw, ph)
    elegidos = []
    for i in candidatos:
        x, y, w, h, area = stats[i]
        cerca = x < px + pw + margen and x + w > px - margen and y < py + ph + margen and y + h > py - margen
        if i == principal or (cerca and area >= 0.08 * parea):
            elegidos.append(i)

    seleccion = np.isin(etiquetas, elegidos).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(seleccion)
    return seleccion, (x, y, w, h)


def ajustar_grosor(recorte):
    """Engrosa los trazos finos para que, al reducir a 20x20, se parezcan a los de MNIST."""
    area = cv2.countNonZero(recorte)
    contornos, _ = cv2.findContours(recorte, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimetro = sum(cv2.arcLength(c, True) for c in contornos)
    if area == 0 or perimetro == 0:
        return recorte
    # Un trazo es casi un rectángulo largo y estrecho: área ≈ longitud x grosor y
    # perímetro ≈ 2 x longitud, así que grosor ≈ 2 x área / perímetro.
    grosor = 2 * area / perimetro
    objetivo = GROSOR_OBJETIVO * max(recorte.shape)
    if grosor < 0.75 * objetivo:
        k = int(round(objetivo - grosor)) | 1  # tamaño impar del engrosado
        nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        recorte = cv2.dilate(recorte, nucleo)  # dilatar = engrosar
    return recorte


def centrar_como_mnist(recorte):
    """Encaja el dígito en 20x20 sin deformarlo y lo centra por su centro de masas en 28x28."""
    alto, ancho = recorte.shape
    factor = LADO_DIGITO / max(alto, ancho)  # el lado mayor pasa a medir 20 píxeles
    nuevo = (max(1, int(round(ancho * factor))), max(1, int(round(alto * factor))))
    reducido = cv2.resize(recorte, nuevo, interpolation=cv2.INTER_AREA)

    # Se pega en el centro de un lienzo negro de 28x28
    lienzo = np.zeros((LADO_SALIDA, LADO_SALIDA), np.float32)
    y0 = (LADO_SALIDA - reducido.shape[0]) // 2
    x0 = (LADO_SALIDA - reducido.shape[1]) // 2
    lienzo[y0 : y0 + reducido.shape[0], x0 : x0 + reducido.shape[1]] = reducido / 255.0

    # Y se mueve para que su centro de masas quede en el centro (14, 14), como en MNIST
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
    """Imagen BGR o en grises (uint8) -> Resultado con la imagen de 28x28 lista para la red.

    Con la webcam o la cámara del móvil, usa fondo="claro": siempre es tinta oscura
    sobre papel. La caja del dígito se devuelve en coordenadas de la imagen original.
    """
    # 0. Las fotos grandes se reducen primero (el resultado es el mismo y va más rápido)
    escala = 1.0
    alto, ancho = imagen.shape[:2]
    if max(alto, ancho) > LADO_MAXIMO:
        escala = LADO_MAXIMO / max(alto, ancho)
        imagen = cv2.resize(imagen, (round(ancho * escala), round(alto * escala)), interpolation=cv2.INTER_AREA)

    # 1 y 2. Grises, suavizado y umbral
    gris = a_gris(imagen)
    mascara = binarizar(gris, fondo)

    # 3. Trazos del dígito
    seleccion, caja = seleccionar_digito(mascara)
    if seleccion is None:
        return Resultado(None, mascara)

    # Recorte alrededor del dígito, con un pequeño margen
    x, y, w, h = caja
    lado = max(w, h)
    margen = int(0.1 * lado) + 2
    recorte = seleccion[max(0, y - margen) : y + h + margen, max(0, x - margen) : x + w + margen]

    # 4. Grosor; tras engrosar, se recorta de nuevo al contenido para que ocupe la caja de 20x20
    recorte = ajustar_grosor(recorte)
    rx, ry, rw, rh = cv2.boundingRect(recorte)
    recorte = recorte[ry : ry + rh, rx : rx + rw]

    # 5. Tamaño y posición de MNIST
    imagen_28 = centrar_como_mnist(recorte)

    pasos = {"gris": gris, "binaria": mascara, "seleccion": seleccion, "recorte": recorte, "entrada_red": imagen_28}
    if escala != 1.0:
        caja = tuple(int(round(v / escala)) for v in caja)  # caja en la imagen original
    return Resultado(imagen_28, mascara, caja, pasos)
