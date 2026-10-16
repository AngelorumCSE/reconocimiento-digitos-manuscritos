"""El preprocesado deja las fotos con el formato de MNIST y no se deja engañar por el fondo."""

import cv2
import numpy as np

from src.preprocesado import LADO_SALIDA, preprocesar, recuadro_central
from src.simulacion import digito_sintetico, dibujo_simulado, foto_simulada, fotograma_simulado


def _comprobar_estilo_mnist(imagen):
    """Comprueba que la imagen es como las de MNIST: 28x28, valores en [0, 1], centrada y en una caja de unos 20x20."""
    assert imagen.shape == (LADO_SALIDA, LADO_SALIDA)
    assert imagen.dtype == np.float32
    assert 0.0 <= imagen.min() and imagen.max() <= 1.0
    momentos = cv2.moments(imagen)
    cx, cy = momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]
    assert abs(cx - 14) < 1.0 and abs(cy - 14) < 1.0  # centrado por centro de masas
    filas, columnas = np.nonzero(imagen > 0.1)
    assert filas.max() - filas.min() <= 22 and columnas.max() - columnas.min() <= 22  # caja de unos 20x20


def test_foto_de_papel():
    """Una foto de un dígito escrito en papel sale con el formato de MNIST."""
    resultado = preprocesar(foto_simulada(digito_sintetico("7"), semilla=3))
    assert resultado.imagen_28 is not None
    _comprobar_estilo_mnist(resultado.imagen_28)


def test_dibujo_sobre_fondo_negro():
    """Un trazo blanco sobre fondo negro (como un dibujo en la web) también funciona."""
    resultado = preprocesar(dibujo_simulado(digito_sintetico("3")))
    assert resultado.imagen_28 is not None
    _comprobar_estilo_mnist(resultado.imagen_28)


def test_imagen_vacia_no_inventa_un_digito():
    """Un papel en blanco con algo de ruido no da ningún dígito."""
    papel = np.full((300, 300, 3), 220, np.uint8)
    papel = cv2.GaussianBlur(papel + np.random.default_rng(0).integers(0, 6, papel.shape, dtype=np.uint8), (3, 3), 0)
    assert preprocesar(papel).imagen_28 is None


def test_ignora_el_borde_del_papel():
    """Una franja oscura pegada al borde de la foto no se toma por parte del dígito."""
    foto = foto_simulada(digito_sintetico("4"), semilla=1)
    foto[:, :12] = 30  # una franja oscura pegada al borde (la mesa, por ejemplo)
    resultado = preprocesar(foto)
    assert resultado.imagen_28 is not None
    assert resultado.caja[0] > 12


def test_hoja_entera_dentro_del_recuadro_de_la_webcam():
    """Si la hoja se ve entera sobre la mesa, su borde no se confunde con un trazo (parecería un 0)."""
    frame = fotograma_simulado(digito_sintetico("5"), semilla=2)
    x, y, lado = recuadro_central(frame)
    resultado = preprocesar(frame[y : y + lado, x : x + lado], fondo="claro")
    assert resultado.imagen_28 is not None
    _, _, w, h = resultado.caja
    assert max(w, h) < 0.5 * lado  # solo el dígito, no el marco de la hoja
    _comprobar_estilo_mnist(resultado.imagen_28)


def test_recuadro_central_es_cuadrado_y_centrado():
    """En un fotograma de 1280x720, el recuadro mide el 60 % del lado menor y está centrado."""
    x, y, lado = recuadro_central(np.zeros((720, 1280, 3), np.uint8))
    assert lado == 432 and (x, y) == (424, 144)


def test_foto_grande_de_movil_da_la_caja_en_coordenadas_originales():
    """Una foto 10 veces más grande da casi la misma imagen de 28x28 y la caja medida en la foto original."""
    pequena = foto_simulada(digito_sintetico("2"), semilla=4)
    grande = cv2.resize(pequena, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)  # 3600 x 3600 px
    r_pequena, r_grande = preprocesar(pequena), preprocesar(grande)
    assert r_pequena.imagen_28 is not None and r_grande.imagen_28 is not None
    assert np.allclose(np.array(r_grande.caja), 10 * np.array(r_pequena.caja), atol=0.04 * grande.shape[0])
    assert np.abs(r_grande.imagen_28 - r_pequena.imagen_28).mean() < 0.1
