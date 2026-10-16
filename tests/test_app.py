"""La web de Streamlit arranca sin errores y tiene sus tres pestañas."""

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(not (RAIZ / "models" / "cnn_aumentada.onnx").exists(), reason="falta el modelo entrenado")
def test_la_web_arranca_sin_errores():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(RAIZ / "app_digitos.py"), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Reconocimiento de dígitos manuscritos"
    assert len(app.tabs) == 3
