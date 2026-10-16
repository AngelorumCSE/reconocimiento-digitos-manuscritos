"""Mide la red con tus propias fotos de dígitos escritos en papel.

Uso:
    python -m src.evaluar_fotos carpeta_con_fotos/

El nombre de cada foto debe empezar por el dígito que contiene: 7.jpg, 7_2.jpg o 3-rotulador.png.
Imprime el resultado de cada foto y la exactitud, guarda un resumen en reports/fotos_reales.json
y una figura con lo que ve la red en reports/figuras/fotos_reales.png (y su versión oscura).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from .graficas import TEMAS, confianza, figura_imagenes
from .inferencia import Clasificador
from .preprocesado import preprocesar

RAIZ = Path(__file__).resolve().parent.parent
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def fotos_con_etiqueta(carpeta):
    """(ruta, dígito real) de cada imagen cuyo nombre empieza por un dígito."""
    fotos, ignoradas = [], []
    for ruta in sorted(Path(carpeta).iterdir()):
        if ruta.suffix.lower() not in EXTENSIONES:
            if ruta.suffix.lower() == ".heic":
                ignoradas.append(f"{ruta.name} (HEIC: conviértela a JPG)")
            continue
        if ruta.name[0].isdigit():
            fotos.append((ruta, int(ruta.name[0])))
        else:
            ignoradas.append(f"{ruta.name} (el nombre no empieza por el dígito)")
    return fotos, ignoradas


def evaluar(carpeta, fondo="claro", carpeta_informe=RAIZ / "reports"):
    fotos, ignoradas = fotos_con_etiqueta(carpeta)
    for texto in ignoradas:
        print(f"  ignorada: {texto}")
    if not fotos:
        raise SystemExit("No hay fotos que empiecen por su dígito (por ejemplo, 7_1.jpg).")

    clasificador = Clasificador()
    detalle, imagenes, rotulos = [], [], []
    for ruta, real in fotos:
        imagen = cv2.imread(str(ruta))
        resultado = preprocesar(imagen, fondo=fondo) if imagen is not None else None
        if resultado is None or resultado.imagen_28 is None:
            detalle.append({"archivo": ruta.name, "real": real, "predicho": None, "confianza": None})
            imagenes.append(np.zeros((28, 28), np.float32))
            rotulos.append(f"{real} → sin dígito")
            print(f"  {ruta.name:<28} real {real}  →  no encuentro el dígito")
            continue
        prob = clasificador.probabilidades(resultado.imagen_28)
        predicho = int(np.argmax(prob))
        detalle.append({"archivo": ruta.name, "real": real, "predicho": predicho, "confianza": float(prob[predicho])})
        imagenes.append(resultado.imagen_28)
        rotulos.append(f"{real} → {predicho} ({confianza(prob[predicho])})")
        marca = "bien" if predicho == real else "FALLO"
        print(f"  {ruta.name:<28} real {real}  →  {predicho}  ({confianza(prob[predicho])})  {marca}")

    n = len(detalle)
    aciertos = sum(d["predicho"] == d["real"] for d in detalle)
    resumen = {
        "fecha": date.today().isoformat(),
        "n": n,
        "aciertos": aciertos,
        "exactitud": aciertos / n,
        "sin_digito": sum(d["predicho"] is None for d in detalle),
        "detalle": detalle,
    }
    carpeta_informe = Path(carpeta_informe)
    carpeta_informe.mkdir(parents=True, exist_ok=True)
    (carpeta_informe / "fotos_reales.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")

    exactitud = f"{100 * aciertos / n:.1f}".replace(".", ",")
    for modo in TEMAS:
        sufijo = "" if modo == "claro" else "_oscuro"
        figura_imagenes(
            imagenes, rotulos, "Mis fotos: lo que ve la red y lo que responde",
            f"Real → predicho (confianza). Acierta {aciertos} de {n} fotos ({exactitud} %).",
            carpeta_informe / "figuras" / f"fotos_reales{sufijo}.png", modo, columnas=min(10, n),
        )
    print(f"\nExactitud con tus fotos: {aciertos} de {n} ({exactitud} %).")
    print(f"Resumen en {carpeta_informe / 'fotos_reales.json'} y figura en {carpeta_informe / 'figuras'}.")
    return resumen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("carpeta", type=Path, help="carpeta con las fotos")
    parser.add_argument("--fondo", choices=["claro", "auto", "oscuro"], default="claro",
                        help="claro (por defecto) = tinta oscura sobre papel")
    args = parser.parse_args()
    evaluar(args.carpeta, args.fondo)


if __name__ == "__main__":
    main()
