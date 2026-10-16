"""Reconocimiento de dígitos en tiempo real con la webcam.

Escribe un dígito en un papel, ponlo dentro del recuadro y la red lo reconoce en
directo. En el panel de la derecha se ve la imagen de 28x28 que recibe la red, el
dígito reconocido y la probabilidad de cada dígito.

Uso:
    python demo_webcam.py                     # webcam por defecto
    python demo_webcam.py --camara 1          # otra cámara
    python demo_webcam.py --imagen foto.jpg   # una foto, sin cámara (guarda foto_reconocido.png)
    python demo_webcam.py --gif demo.gif      # graba un GIF mientras la usas

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

# Colores en BGR (el orden de OpenCV). El azul es el mismo de las figuras del README.
AZUL = (214, 120, 42)
BLANCO = (245, 245, 245)
GRIS = (150, 150, 150)
GRIS_OSCURO = (48, 48, 47)
FONDO = (25, 25, 26)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX  # las fuentes de OpenCV no tienen tildes: los textos van sin ellas

ALTO_VISTA = 540  # alto de la ventana: cabe en la pantalla de un portátil
ANCHO_PANEL = 330  # ancho del panel de la derecha
MARGEN = 20
GIF_ESCALA = 0.6  # el GIF se guarda más pequeño que la ventana para que pese poco
GIF_PASO = 0.1  # segundos entre fotogramas del GIF (10 por segundo)
GIF_MAXIMO = 20.0  # segundos como máximo


def texto(imagen, contenido, posicion, escala=0.5, color=BLANCO, grosor=1):
    """Escribe un texto en la imagen (con bordes suavizados)."""
    cv2.putText(imagen, contenido, posicion, FUENTE, escala, color, grosor, cv2.LINE_AA)


def panel_lateral(resultado, prob, umbral, fps=None, con_teclas=False):
    """Panel con lo que ve la red (28x28), lo que decide y la probabilidad de cada dígito."""
    panel = np.full((ALTO_VISTA, ANCHO_PANEL, 3), FONDO, np.uint8)
    m = MARGEN

    # 1. La imagen de 28x28 que recibe la red, ampliada a 140x140
    texto(panel, "Entrada de la red (28x28)", (m, 32), color=GRIS)
    if resultado is not None and resultado.imagen_28 is not None:
        entrada = cv2.resize((resultado.imagen_28 * 255).astype(np.uint8), (140, 140), interpolation=cv2.INTER_NEAREST)
        panel[44:184, m : m + 140] = cv2.cvtColor(entrada, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(panel, (m - 1, 43), (m + 140, 184), GRIS, 1)

    # 2. El dígito reconocido (o «?» si la red no está segura)
    if prob is not None:
        mejor = int(np.argmax(prob))
        confianza = float(prob[mejor])
        seguro = confianza >= umbral
        texto(panel, str(mejor) if seguro else "?", (m + 170, 150), escala=4.0, grosor=8)
        texto(panel, f"{100 * confianza:.0f}% seguro" if seguro else "poco seguro", (m + 168, 180))
    else:
        texto(panel, "Sin digito", (m + 168, 120), escala=0.6, color=GRIS)

    # 3. Una barra por dígito con su probabilidad
    y0 = 232
    texto(panel, "Probabilidad de cada digito", (m, y0 - 14), color=GRIS)
    ancho_barra = ANCHO_PANEL - 2 * m - 20 - 48
    for d in range(10):
        yb = y0 + d * 24
        texto(panel, str(d), (m, yb + 14))
        cv2.rectangle(panel, (m + 20, yb), (m + 20 + ancho_barra, yb + 16), GRIS_OSCURO, -1)  # fondo de la barra
        if prob is not None:
            largo = int(round(float(prob[d]) * ancho_barra))
            if largo > 0:
                color = AZUL if d == int(np.argmax(prob)) else GRIS
                cv2.rectangle(panel, (m + 20, yb), (m + 20 + largo, yb + 16), color, -1)
            texto(panel, f"{100 * prob[d]:.0f}%", (m + 28 + ancho_barra, yb + 14), escala=0.45)

    # 4. Pie: fotogramas por segundo y teclas
    if fps is not None:
        texto(panel, f"{fps:.0f} FPS  |  ONNX Runtime", (m, ALTO_VISTA - 42), escala=0.45, color=GRIS)
    if con_teclas:
        texto(panel, "q: salir    s: guardar captura", (m, ALTO_VISTA - 18), escala=0.45, color=GRIS)
    return panel


def componer(frame, resultado, prob, umbral, recuadro=None, fps=None):
    """La imagen de la cámara (o la foto) a la izquierda y el panel de la red a la derecha."""
    # Se escala la imagen para que mida ALTO_VISTA de alto
    escala = ALTO_VISTA / frame.shape[0]
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    vista = cv2.resize(frame, (int(round(frame.shape[1] * escala)), ALTO_VISTA), interpolation=interpolacion)

    def a_vista(x, y):
        """De coordenadas de la imagen original a coordenadas de la vista."""
        return int(round(x * escala)), int(round(y * escala))

    dx = dy = 0
    if recuadro is not None:  # recuadro blanco donde hay que poner el dígito
        x0, y0, lado = recuadro
        esquina = a_vista(x0, y0)
        cv2.rectangle(vista, esquina, a_vista(x0 + lado, y0 + lado), BLANCO, 2, cv2.LINE_AA)
        texto(vista, "Coloca el digito dentro del recuadro", (esquina[0], max(esquina[1] - 12, 22)), escala=0.55)
        dx, dy = x0, y0  # la caja del dígito está medida dentro del recuadro
    if resultado is not None and resultado.imagen_28 is not None:  # caja azul alrededor del dígito
        x, y, w, h = resultado.caja
        cv2.rectangle(vista, a_vista(dx + x, dy + y), a_vista(dx + x + w, dy + y + h), AZUL, 2, cv2.LINE_AA)

    panel = panel_lateral(resultado, prob, umbral, fps, con_teclas=recuadro is not None)
    return np.hstack([vista, panel])  # una imagen al lado de la otra


def modo_imagen(ruta, clasificador, umbral, fondo="auto"):
    """Reconoce el dígito de una foto y guarda la imagen anotada junto a ella."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        raise SystemExit(f"No puedo abrir la imagen {ruta}")
    resultado = preprocesar(imagen, fondo=fondo)
    if resultado.imagen_28 is None:
        print("No encuentro ningún dígito. Prueba con más luz, un trazo más grueso o el dígito más grande.")
        return None
    prob = clasificador.probabilidades(resultado.imagen_28)
    for d in np.argsort(-prob)[:3]:  # los tres dígitos más probables
        print(f"  {d}: {100 * prob[d]:.1f} %")
    destino = Path(ruta).with_name(Path(ruta).stem + "_reconocido.png")
    cv2.imwrite(str(destino), componer(imagen, resultado, prob, umbral))
    print(f"Imagen anotada guardada en {destino}")
    return destino


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
    """Bucle principal: lee la cámara, reconoce el dígito y enseña el resultado en una ventana."""
    camara = cv2.VideoCapture(indice)
    if not camara.isOpened():
        raise SystemExit(
            "No puedo abrir la cámara. En macOS, da permiso de cámara a la Terminal "
            "(Ajustes del Sistema > Privacidad y seguridad > Cámara)."
        )
    historial = deque(maxlen=5)  # últimas 5 predicciones: su media evita que el resultado parpadee
    tiempo_anterior = time.perf_counter()
    fps = 0.0
    fotogramas_gif, ultimo_gif = [], 0.0
    print("Pulsa q para salir y s para guardar una captura.")
    if gif:
        print(f"Grabando el GIF (hasta {GIF_MAXIMO:.0f} s). Pulsa q para terminar y guardarlo.")

    while True:
        ok, frame = camara.read()
        if not ok:
            break

        # El dígito solo se busca dentro del recuadro central
        x0, y0, lado = recuadro = recuadro_central(frame)
        resultado = preprocesar(frame[y0 : y0 + lado, x0 : x0 + lado], fondo="claro")
        if resultado.imagen_28 is not None:
            historial.append(clasificador.probabilidades(resultado.imagen_28))
        else:
            historial.clear()
        prob = np.mean(historial, axis=0) if historial else None

        # Fotogramas por segundo (suavizados para que el número no baile)
        ahora = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(ahora - tiempo_anterior, 1e-6)
        tiempo_anterior = ahora

        vista = componer(frame, resultado, prob, umbral, recuadro=recuadro, fps=fps)
        cv2.imshow("Reconocimiento de digitos (q para salir)", vista)
        if gif and ahora - ultimo_gif >= GIF_PASO and len(fotogramas_gif) < GIF_MAXIMO / GIF_PASO:
            fotogramas_gif.append(cv2.resize(vista, None, fx=GIF_ESCALA, fy=GIF_ESCALA, interpolation=cv2.INTER_AREA))
            ultimo_gif = ahora

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), 27):  # q o Esc
            break
        if tecla == ord("s"):
            carpeta = Path("capturas")
            carpeta.mkdir(exist_ok=True)
            nombre = carpeta / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(nombre), vista)
            print(f"Captura guardada en {nombre}")

    camara.release()
    cv2.destroyAllWindows()
    if gif:
        guardar_gif(fotogramas_gif, gif)


def main():
    """Lee las opciones de la línea de órdenes y arranca el modo elegido."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camara", type=int, default=0, help="índice de la cámara (0 por defecto)")
    parser.add_argument("--imagen", type=Path, help="analizar una foto en lugar de la webcam")
    parser.add_argument("--umbral", type=float, default=0.6, help="confianza mínima para mostrar el dígito")
    parser.add_argument(
        "--fondo", choices=["auto", "claro", "oscuro"], default="auto",
        help="solo con --imagen: claro = tinta sobre papel, oscuro = trazo claro sobre negro",
    )
    parser.add_argument("--gif", type=Path, metavar="RUTA", help="grabar un GIF de la demo (se guarda al pulsar q)")
    args = parser.parse_args()

    clasificador = Clasificador()
    if args.imagen:
        modo_imagen(args.imagen, clasificador, args.umbral, args.fondo)
    else:
        modo_webcam(args.camara, clasificador, args.umbral, args.gif)


if __name__ == "__main__":
    main()
