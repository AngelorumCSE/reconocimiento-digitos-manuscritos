"""Prueba completa: foto simulada -> preprocesado -> red, con dígitos reales del test de MNIST."""

from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
HAY_MODELO = (RAIZ / "models" / "cnn_aumentada.onnx").exists() or (RAIZ / "models" / "cnn_aumentada.pt").exists()
HAY_DATOS = (RAIZ / "data" / "MNIST" / "raw").exists()


@pytest.mark.skipif(not (HAY_MODELO and HAY_DATOS), reason="faltan el modelo entrenado o MNIST")
def test_reconoce_fotos_simuladas_de_digitos_del_test():
    from torchvision import datasets

    from src.inferencia import Clasificador
    from src.preprocesado import preprocesar
    from src.simulacion import foto_simulada

    test = datasets.MNIST(RAIZ / "data", train=False, download=False)
    clasificador = Clasificador()
    aciertos, n = 0, 200
    for i in range(n):
        imagen = test.data[i].numpy().astype(np.float32) / 255.0
        resultado = preprocesar(foto_simulada(imagen, semilla=i))
        if resultado.imagen_28 is not None:
            aciertos += int(np.argmax(clasificador.probabilidades(resultado.imagen_28)) == int(test.targets[i]))
    assert aciertos / n >= 0.95


@pytest.mark.skipif(not (HAY_MODELO and HAY_DATOS), reason="faltan el modelo entrenado o MNIST")
def test_reconoce_fotogramas_simulados_de_la_webcam():
    # hoja entera dentro del recuadro, sobre una mesa oscura y algo girada, como en la demo
    from torchvision import datasets

    from src.inferencia import Clasificador
    from src.preprocesado import preprocesar, recuadro_central
    from src.simulacion import fotograma_simulado

    test = datasets.MNIST(RAIZ / "data", train=False, download=False)
    clasificador = Clasificador()
    aciertos, n = 0, 100
    for i in range(n):
        imagen = test.data[i].numpy().astype(np.float32) / 255.0
        frame = fotograma_simulado(imagen, semilla=i)
        x, y, lado = recuadro_central(frame)
        resultado = preprocesar(frame[y : y + lado, x : x + lado], fondo="claro")
        if resultado.imagen_28 is not None:
            aciertos += int(np.argmax(clasificador.probabilidades(resultado.imagen_28)) == int(test.targets[i]))
    assert aciertos / n >= 0.95
