from pathlib import Path

import numpy as np
import pytest
import torch

from src.entrenamiento import aumentar_lote
from src.modelos import crear_modelo

RAIZ = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("nombre", ["mlp", "lenet", "cnn_aumentada"])
def test_forma_de_la_salida(nombre):
    modelo = crear_modelo(nombre).eval()
    with torch.no_grad():
        salida = modelo(torch.zeros(3, 1, 28, 28))
    assert salida.shape == (3, 10)


def test_aumento_mantiene_forma_y_rango():
    x = torch.rand(8, 1, 28, 28)
    y = aumentar_lote(x.clone(), torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0 + 1e-6


@pytest.mark.skipif(not (RAIZ / "models" / "cnn_aumentada.onnx").exists(), reason="falta el modelo entrenado")
def test_onnx_y_pytorch_dan_lo_mismo():
    import onnxruntime as ort

    modelo = crear_modelo("cnn_aumentada")
    modelo.load_state_dict(torch.load(RAIZ / "models" / "cnn_aumentada.pt", map_location="cpu"))
    modelo.eval()
    x = np.random.default_rng(0).random((4, 1, 28, 28)).astype(np.float32)
    with torch.no_grad():
        ref = modelo(torch.from_numpy(x)).numpy()
    sesion = ort.InferenceSession(str(RAIZ / "models" / "cnn_aumentada.onnx"), providers=["CPUExecutionProvider"])
    assert np.abs(sesion.run(None, {"imagen": x})[0] - ref).max() < 1e-4
