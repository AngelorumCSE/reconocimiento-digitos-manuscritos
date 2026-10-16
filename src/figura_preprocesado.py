"""Regenera la figura del preprocesado con una foto real tuya.

Uso:
    python -m src.figura_preprocesado foto.jpg

Sustituye reports/figuras/preprocesado.png e imprime el dígito reconocido, para que
el README enseñe un ejemplo real en lugar del simulado.
"""

import sys

import cv2
import numpy as np

from .graficas import FIGURAS, figura_pasos, frase_confianza
from .inferencia import Clasificador
from .preprocesado import preprocesar


def main():
    """Preprocesa la foto indicada, la clasifica y dibuja los pasos."""
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    foto = cv2.imread(sys.argv[1])
    if foto is None:
        raise SystemExit(f"No puedo abrir {sys.argv[1]}")
    resultado = preprocesar(foto)
    if resultado.imagen_28 is None:
        raise SystemExit("No encuentro ningún dígito en la foto: prueba con más luz o un trazo más grueso.")

    prob = Clasificador().probabilidades(resultado.imagen_28)
    digito = int(np.argmax(prob))
    pasos = {**resultado.pasos, "gris": foto}  # en el primer panel, la foto original en color
    subtitulo = f"Foto real. La red reconoce un {digito} {frase_confianza(prob[digito])}."
    figura_pasos(pasos, FIGURAS / "preprocesado.png", subtitulo)
    print(f"Dígito reconocido: {digito} ({100 * prob[digito]:.1f} %). Figura actualizada en {FIGURAS}.")


if __name__ == "__main__":
    main()
