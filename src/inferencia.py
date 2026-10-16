"""Carga la red entrenada (exportada a ONNX) para reconocer un dígito.

La usan la web, la demo con webcam y la evaluación con fotos. Con ONNX Runtime no
hace falta instalar PyTorch, que ocupa mucho más: por eso la web es tan ligera.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "cnn_aumentada.onnx"


class Clasificador:
    """Envuelve la red exportada a ONNX: se crea una vez y se usa con todas las imágenes."""

    def __init__(self, ruta=RUTA_MODELO):
        """Abre el modelo (por defecto, models/cnn_aumentada.onnx)."""
        self.sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])

    def probabilidades(self, imagen_28):
        """Probabilidad de cada dígito (array de 10 valores que suman 1) para una imagen 28x28 en [0, 1]."""
        x = np.asarray(imagen_28, dtype=np.float32)[None, None]  # la red espera la forma 1 x 1 x 28 x 28
        logits = self.sesion.run(None, {"imagen": x})[0][0]  # una puntuación por dígito
        # Softmax: convierte las puntuaciones en probabilidades (restar el máximo evita desbordes)
        e = np.exp(logits - logits.max())
        return e / e.sum()
