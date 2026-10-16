"""Regenera la figura del preprocesado con una foto real tuya.

Uso:
    python -m src.figura_preprocesado foto.jpg

Sustituye reports/figuras/preprocesado.png y preprocesado_oscuro.png e imprime el
dígito reconocido, para que el README enseñe un ejemplo real en lugar del simulado.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from .graficas import TEMAS, figura_pasos, frase_confianza
from .inferencia import Clasificador
from .preprocesado import preprocesar

FIGURAS = Path(__file__).resolve().parent.parent / "reports" / "figuras"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    foto = cv2.imread(argv[0])
    if foto is None:
        raise SystemExit(f"No puedo abrir {argv[0]}")
    resultado = preprocesar(foto)
    if resultado.imagen_28 is None:
        raise SystemExit("No encuentro ningún dígito en la foto: prueba con más luz o un trazo más grueso.")

    prob = Clasificador().probabilidades(resultado.imagen_28)
    digito = int(np.argmax(prob))
    pasos = dict(resultado.pasos)
    pasos["gris"] = foto
    subtitulo = f"Foto real. La red reconoce un {digito} {frase_confianza(prob[digito])}."
    for modo in TEMAS:
        sufijo = "" if modo == "claro" else "_oscuro"
        figura_pasos(pasos, FIGURAS / f"preprocesado{sufijo}.png", modo, subtitulo=subtitulo)
    print(f"Dígito reconocido: {digito} ({100 * prob[digito]:.1f} %). Figura actualizada en {FIGURAS}.")


if __name__ == "__main__":
    main()
