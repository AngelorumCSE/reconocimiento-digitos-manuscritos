"""La evaluación con fotos propias: lee el dígito del nombre, mide y guarda el resumen y la figura."""

import json
from pathlib import Path

import cv2
import pytest

from src.simulacion import digito_sintetico, foto_simulada

RAIZ = Path(__file__).resolve().parent.parent
HAY_MODELO = (RAIZ / "models" / "cnn_aumentada.onnx").exists()


@pytest.mark.skipif(not HAY_MODELO, reason="falta el modelo entrenado")
def test_evalua_una_carpeta_de_fotos(tmp_path):
    """Con 4 fotos con etiqueta, una sin etiqueta y un .txt, evalúa solo las 4 y guarda el resumen y la figura."""
    from src.evaluar_fotos import evaluar

    fotos = tmp_path / "fotos"
    fotos.mkdir()
    for i, digito in enumerate("0347"):
        cv2.imwrite(str(fotos / f"{digito}_{i}.jpg"), foto_simulada(digito_sintetico(digito), semilla=i))
    (fotos / "notas.txt").write_text("no es una imagen")
    cv2.imwrite(str(fotos / "sin_etiqueta.jpg"), foto_simulada(digito_sintetico("5")))

    resumen = evaluar(fotos, carpeta_informe=tmp_path / "informe")
    assert resumen["n"] == 4  # la foto sin dígito en el nombre y el .txt se ignoran
    assert {d["real"] for d in resumen["detalle"]} == {0, 3, 4, 7}
    guardado = json.loads((tmp_path / "informe" / "fotos_reales.json").read_text(encoding="utf-8"))
    assert guardado["aciertos"] == resumen["aciertos"]
    assert (tmp_path / "informe" / "figuras" / "fotos_reales.png").exists()
