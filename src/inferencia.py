"""Carga de la red entrenada para la web y la demo.

Usa ONNX Runtime si está instalado (más ligero, ideal para desplegar) y, si no,
PyTorch con los pesos guardados. Las dos opciones dan el mismo resultado.
"""

from pathlib import Path

import numpy as np

RUTA_MODELOS = Path(__file__).resolve().parent.parent / "models"


class Clasificador:
    def __init__(self, carpeta=RUTA_MODELOS):
        carpeta = Path(carpeta)
        try:
            import onnxruntime as ort

            self._sesion = ort.InferenceSession(str(carpeta / "cnn_aumentada.onnx"), providers=["CPUExecutionProvider"])
            self.motor = "ONNX Runtime"
        except ImportError:
            import torch

            from .modelos import crear_modelo

            self._modelo = crear_modelo("cnn_aumentada")
            self._modelo.load_state_dict(torch.load(carpeta / "cnn_aumentada.pt", map_location="cpu"))
            self._modelo.eval()
            self._sesion = None
            self.motor = "PyTorch"

    def probabilidades(self, imagen_28):
        """Probabilidad de cada dígito (array de 10) para una imagen 28x28 en [0, 1]."""
        x = np.asarray(imagen_28, dtype=np.float32)[None, None]
        if self._sesion is not None:
            logits = self._sesion.run(None, {"imagen": x})[0][0]
        else:
            import torch

            with torch.no_grad():
                logits = self._modelo(torch.from_numpy(x))[0].numpy()
        e = np.exp(logits - logits.max())
        return e / e.sum()
