"""Reconocimiento de dígitos en tiempo real con la webcam.

Escribe un dígito en un papel, ponlo dentro del recuadro y la red lo reconoce en
directo. En el panel de la derecha verás la imagen de 28x28 que recibe la red, el
dígito reconocido y la probabilidad de cada dígito.

Uso:
    python demo_webcam.py                     # webcam por defecto
    python demo_webcam.py --camara 1          # otra cámara
    python demo_webcam.py --imagen foto.jpg   # una foto, sin cámara (guarda foto_reconocido.png)
    python demo_webcam.py --gif reports/figuras/demo_webcam.gif   # graba un GIF mientras la usas
    python demo_webcam.py --captura-simulada reports/figuras/demo_webcam.png   # captura de ejemplo

Teclas: q o Esc para salir, s para guardar una captura en la carpeta capturas/.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.inferencia import Clasificador
from src.preprocesado import preprocesar, recuadro_central

# Colores en BGR (OpenCV). El azul es el mismo de las figuras del README (#2a78d6).
AZUL = (214, 120, 42)
BLANCO = (245, 245, 245)
GRIS = (150, 150, 150)
PISTA = (48, 48, 47)
FONDO = (25, 25, 26)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX  # las fuentes de OpenCV no tienen tildes: los textos van sin ellas

ALTO_VISTA = 540  # alto de la ventana: cabe en la pantalla de un portátil
GIF_ESCALA = 0.6  # el GIF se guarda más pequeño que la ventana para que pese poco
GIF_PASO = 0.1  # segundos entre fotogramas del GIF (10 por segundo, se reproduce a velocidad real)
GIF_MAXIMO = 20.0  # segundos como máximo
ANCHO_PANEL = 330
MARGEN = 20


def _texto(imagen, texto, posicion, escala=0.5, color=BLANCO, grosor=1):
    cv2.putText(imagen, texto, posicion, FUENTE, escala, color, grosor, cv2.LINE_AA)


def _panel_lateral(resultado, prob, umbral, fps=None, motor=None, con_teclas=False):
    """Panel con lo que ve la red (28x28), lo que decide y la probabilidad de cada dígito."""
    panel = np.full((ALTO_VISTA, ANCHO_PANEL, 3), FONDO, np.uint8)
    m = MARGEN

    _texto(panel, "Entrada de la red (28x28)", (m, 32), color=GRIS)
    if resultado is not None and resultado.imagen_28 is not None:
        entrada = cv2.resize((resultado.imagen_28 * 255).astype(np.uint8), (140, 140), interpolation=cv2.INTER_NEAREST)
        panel[44:184, m : m + 140] = cv2.cvtColor(entrada, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (m - 1, 43), (m + 140, 184), GRIS, 1)

    if prob is not None:
        mejor = int(np.argmax(prob))
        confianza = float(prob[mejor])
        seguro = confianza >= umbral
        _texto(panel, str(mejor) if seguro else "?", (m + 170, 150), escala=4.0, grosor=8)
        _texto(panel, f"{100 * confianza:.0f}% seguro" if seguro else "poco seguro", (m + 168, 180))
    else:
        _texto(panel, "Sin digito", (m + 168, 120), escala=0.6, color=GRIS)

    y0 = 232
    _texto(panel, "Probabilidad de cada digito", (m, y0 - 14), color=GRIS)
    ancho_barra = ANCHO_PANEL - 2 * m - 20 - 48
    for d in range(10):
        yb = y0 + d * 24
        _texto(panel, str(d), (m, yb + 14))
        cv2.rectangle(panel, (m + 20, yb), (m + 20 + ancho_barra, yb + 16), PISTA, -1)
        if prob is not None:
            largo = int(round(float(prob[d]) * ancho_barra))
            if largo > 0:
                color = AZUL if d == int(np.argmax(prob)) else GRIS
                cv2.rectangle(panel, (m + 20, yb), (m + 20 + largo, yb + 16), color, -1)
            _texto(panel, f"{100 * prob[d]:.0f}%", (m + 28 + ancho_barra, yb + 14), escala=0.45)

    pie = [p for p in (f"{fps:.0f} FPS" if fps is not None else None, motor) if p]
    if pie:
        _texto(panel, "  |  ".join(pie), (m, ALTO_VISTA - 42), escala=0.45, color=GRIS)
    if con_teclas:
        _texto(panel, "q: salir    s: guardar captura", (m, ALTO_VISTA - 18), escala=0.45, color=GRIS)
    return panel


def componer(frame, resultado, prob, umbral, roi=None, fps=None, motor=None):
    """La imagen de la cámara (o la foto) a la izquierda y el panel de la red a la derecha, sin taparse."""
    escala = ALTO_VISTA / frame.shape[0]
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    vista = cv2.resize(frame, (int(round(frame.shape[1] * escala)), ALTO_VISTA), interpolation=interpolacion)

    def a_vista(x, y):
        return int(round(x * escala)), int(round(y * escala))

    dx = dy = 0
    if roi is not None:
        x0, y0, lado = roi
        esquina = a_vista(x0, y0)
        cv2.rectangle(vista, esquina, a_vista(x0 + lado, y0 + lado), BLANCO, 2, cv2.LINE_AA)
        _texto(vista, "Coloca el digito dentro del recuadro", (esquina[0], max(esquina[1] - 12, 22)), escala=0.55)
        dx, dy = x0, y0
    if resultado is not None and resultado.imagen_28 is not None:
        x, y, w, h = resultado.caja
        cv2.rectangle(vista, a_vista(dx + x, dy + y), a_vista(dx + x + w, dy + y + h), AZUL, 2, cv2.LINE_AA)

    panel = _panel_lateral(resultado, prob, umbral, fps, motor, con_teclas=roi is not None)
    return np.hstack([vista, panel])


def modo_imagen(ruta, clasificador, umbral, fondo="auto"):
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        raise SystemExit(f"No puedo abrir la imagen {ruta}")
    resultado = preprocesar(imagen, fondo=fondo)
    if resultado.imagen_28 is None:
        print("No encuentro ningún dígito. Prueba con más luz, un trazo más grueso o el dígito más grande.")
        return None
    prob = clasificador.probabilidades(resultado.imagen_28)
    for d in np.argsort(-prob)[:3]:
        print(f"  {d}: {100 * prob[d]:.1f} %")
    salida = componer(imagen, resultado, prob, umbral, motor=clasificador.motor)
    destino = Path(ruta).with_name(Path(ruta).stem + "_reconocido.png")
    cv2.imwrite(str(destino), salida)
    print(f"Imagen anotada guardada en {destino}")
    return destino


def captura_simulada(destino, clasificador, umbral, indice=1, semilla=11):
    """Guarda una captura de la demo hecha con un fotograma simulado (un dígito del test de MNIST)."""
    from src.datos import cargar_mnist
    from src.simulacion import fotograma_simulado

    frame = fotograma_simulado(cargar_mnist()["X_test"][indice], semilla=semilla)
    x0, y0, lado = roi = recuadro_central(frame)
    resultado = preprocesar(frame[y0 : y0 + lado, x0 : x0 + lado], fondo="claro")
    prob = clasificador.probabilidades(resultado.imagen_28) if resultado.imagen_28 is not None else None
    cv2.imwrite(str(destino), componer(frame, resultado, prob, umbral, roi=roi, motor=clasificador.motor))
    print(f"Captura simulada guardada en {destino}")


def guardar_gif(fotogramas, destino):
    """Guarda los fotogramas (BGR) como un GIF animado que se repite."""
    from PIL import Image

    if not fotogramas:
        print("No se ha grabado ningún fotograma.")
        return
    imagenes = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for f in fotogramas
    ]
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    imagenes[0].save(
        destino, save_all=True, append_images=imagenes[1:], duration=int(GIF_PASO * 1000), loop=0, optimize=True
    )
    megas = f"{Path(destino).stat().st_size / 1e6:.1f}".replace(".", ",")
    print(f"GIF guardado en {destino} ({len(imagenes)} fotogramas, {megas} MB)")


def modo_webcam(indice, clasificador, umbral, gif=None):
    camara = cv2.VideoCapture(indice)
    if not camara.isOpened():
        raise SystemExit(
            "No puedo abrir la cámara. En macOS, da permiso de cámara a la Terminal "
            "(Ajustes del Sistema > Privacidad y seguridad > Cámara)."
        )
    historial = deque(maxlen=5)  # promedio de los últimos fotogramas para que la predicción no parpadee
    carpeta = Path("capturas")
    tiempo_anterior = time.perf_counter()
    fps = 0.0
    print(f"Motor de inferencia: {clasificador.motor}. Pulsa q para salir, s para guardar una captura.")
    fotogramas_gif, ultimo_gif = [], 0.0
    if gif:
        print(f"Grabando el GIF (hasta {GIF_MAXIMO:.0f} s). Pulsa q para terminar y guardarlo.")

    while True:
        ok, frame = camara.read()
        if not ok:
            break
        x0, y0, lado = roi = recuadro_central(frame)
        resultado = preprocesar(frame[y0 : y0 + lado, x0 : x0 + lado], fondo="claro")
        if resultado.imagen_28 is not None:
            historial.append(clasificador.probabilidades(resultado.imagen_28))
        else:
            historial.clear()
        prob = np.mean(historial, axis=0) if historial else None

        ahora = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(ahora - tiempo_anterior, 1e-6)
        tiempo_anterior = ahora
        vista = componer(frame, resultado, prob, umbral, roi=roi, fps=fps, motor=clasificador.motor)
        cv2.imshow("Reconocimiento de digitos (q para salir)", vista)
        if gif and ahora - ultimo_gif >= GIF_PASO and len(fotogramas_gif) < GIF_MAXIMO / GIF_PASO:
            fotogramas_gif.append(cv2.resize(vista, None, fx=GIF_ESCALA, fy=GIF_ESCALA, interpolation=cv2.INTER_AREA))
            ultimo_gif = ahora

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):
            break
        if tecla == ord("s"):
            carpeta.mkdir(exist_ok=True)
            nombre = carpeta / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(nombre), vista)
            print(f"Captura guardada en {nombre}")

    camara.release()
    cv2.destroyAllWindows()
    if gif:
        guardar_gif(fotogramas_gif, gif)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camara", type=int, default=0, help="índice de la cámara (0 por defecto)")
    parser.add_argument("--imagen", type=Path, help="analizar una foto en lugar de la webcam")
    parser.add_argument("--umbral", type=float, default=0.6, help="confianza mínima para mostrar el dígito")
    parser.add_argument(
        "--fondo", choices=["auto", "claro", "oscuro"], default="auto",
        help="solo con --imagen: claro = tinta sobre papel, oscuro = trazo claro sobre negro",
    )
    parser.add_argument("--gif", type=Path, metavar="RUTA", help="grabar un GIF de la demo (se guarda al pulsar q)")
    parser.add_argument("--captura-simulada", type=Path, metavar="RUTA", help="guardar una captura con un fotograma simulado")
    args = parser.parse_args()

    clasificador = Clasificador()
    if args.captura_simulada:
        captura_simulada(args.captura_simulada, clasificador, args.umbral)
    elif args.imagen:
        modo_imagen(args.imagen, clasificador, args.umbral, args.fondo)
    else:
        modo_webcam(args.camara, clasificador, args.umbral, args.gif)


if __name__ == "__main__":
    main()
