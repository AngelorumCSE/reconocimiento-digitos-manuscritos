"""La demo con webcam, sin cámara: un vídeo simulado sustituye a la webcam y a la ventana."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.simulacion import digito_sintetico, fotograma_simulado

RAIZ = Path(__file__).resolve().parent.parent
HAY_MODELO = (RAIZ / "models" / "cnn_aumentada.onnx").exists()


class CamaraSimulada:
    """Imita cv2.VideoCapture: devuelve unos cuantos fotogramas con un dígito y luego se acaba."""

    def __init__(self, n=12):
        """Prepara n fotogramas simulados de webcam con un 3 escrito en una hoja."""
        self.fotogramas = [fotograma_simulado(digito_sintetico("3"), 640, 480, semilla=i) for i in range(n)]

    def isOpened(self):
        """La cámara simulada siempre está disponible."""
        return True

    def read(self):
        """Devuelve el siguiente fotograma, o (False, None) cuando ya no quedan."""
        if not self.fotogramas:
            return False, None
        return True, self.fotogramas.pop(0)

    def release(self):
        """No hay nada que cerrar."""
        pass


@pytest.mark.skipif(not HAY_MODELO, reason="falta el modelo entrenado")
def test_demo_procesa_fotogramas_y_graba_el_gif(tmp_path, monkeypatch):
    """La demo enseña los 12 fotogramas con el panel lateral y guarda un GIF que no está vacío."""
    import demo_webcam

    mostrados = []
    monkeypatch.setattr(demo_webcam.cv2, "VideoCapture", lambda indice: CamaraSimulada())
    monkeypatch.setattr(demo_webcam.cv2, "imshow", lambda titulo, imagen: mostrados.append(imagen))
    monkeypatch.setattr(demo_webcam.cv2, "waitKey", lambda ms: -1)
    monkeypatch.setattr(demo_webcam.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(demo_webcam, "GIF_PASO", 1e-9)  # guarda todos los fotogramas

    destino = tmp_path / "demo.gif"
    demo_webcam.modo_webcam(0, demo_webcam.Clasificador(), 0.6, gif=destino)

    assert len(mostrados) == 12
    assert mostrados[0].shape[0] == demo_webcam.ALTO_VISTA  # vista de la cámara + panel lateral
    assert mostrados[0].shape[1] == int(round(640 * demo_webcam.ALTO_VISTA / 480)) + demo_webcam.ANCHO_PANEL
    gif = cv2.VideoCapture(str(destino))
    ok, primero = gif.read()
    assert destino.exists() and ok and primero is not None
    assert np.mean(primero) > 10  # no es un GIF vacío
