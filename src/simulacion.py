"""Imágenes sintéticas para probar el preprocesado sin cámara.

- `foto_simulada`: foto de cerca de un dígito escrito con bolígrafo en un papel (fondo
  claro con sombra, tinta oscura, ruido, desenfoque y un pequeño giro).
- `fotograma_simulado`: fotograma de webcam con la hoja entera, algo girada, sobre una mesa.
- `dibujo_simulado`: dígito dibujado con el ratón (trazo blanco y grueso sobre fondo negro).
- `digito_sintetico`: un dígito de 28x28 escrito con una fuente, para pruebas sin MNIST.

Sirven para los tests y para las figuras; para una prueba real, usa la demo con la webcam,
`--imagen` con una foto tuya o `python -m src.evaluar_fotos` con una carpeta de fotos.
"""

import cv2
import numpy as np


def digito_sintetico(texto="7"):
    """Un dígito de 28x28 dibujado con una fuente de OpenCV (para pruebas que no necesitan MNIST)."""
    lienzo = np.zeros((28, 28), np.uint8)
    cv2.putText(lienzo, texto, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    return lienzo.astype(np.float32) / 255.0


def foto_simulada(imagen_28, lado=360, semilla=0):
    rng = np.random.default_rng(semilla)
    tam = lado // 2
    digito = cv2.resize(imagen_28.astype(np.float32), (tam, tam), interpolation=cv2.INTER_CUBIC)
    digito = np.clip(digito, 0, 1)

    tinta = np.zeros((lado, lado), np.float32)
    dx, dy = (int(v) for v in rng.integers(-lado // 10, lado // 10, size=2))
    y0, x0 = lado // 4 + dy, lado // 4 + dx
    tinta[y0 : y0 + tam, x0 : x0 + tam] = digito
    giro = cv2.getRotationMatrix2D((lado / 2, lado / 2), float(rng.uniform(-8, 8)), 1.0)
    tinta = cv2.warpAffine(tinta, giro, (lado, lado))

    filas = np.linspace(0, 1, lado, dtype=np.float32)[:, None]
    columnas = np.linspace(0, 1, lado, dtype=np.float32)[None, :]
    luz = 0.80 + 0.15 * columnas - 0.12 * filas  # luz lateral y sombra en la parte baja
    papel = np.stack([np.full((lado, lado), c, np.float32) for c in (210, 222, 228)], axis=-1) / 255.0
    papel = papel * luz[..., None]

    color_tinta = np.array([70, 40, 30], np.float32) / 255.0  # bolígrafo azul oscuro (BGR)
    cobertura = np.clip(tinta * 1.3, 0, 1)[..., None]
    foto = papel * (1 - cobertura) + color_tinta * cobertura
    foto += rng.normal(0, 0.02, foto.shape).astype(np.float32)
    foto = cv2.GaussianBlur(foto, (3, 3), 0)
    return (np.clip(foto, 0, 1) * 255).astype(np.uint8)


def fotograma_simulado(imagen_28, ancho=1280, alto=720, fraccion_hoja=0.45, semilla=0):
    """Fotograma de webcam: una hoja con el dígito, algo girada, sobre una mesa más oscura.

    Con fraccion_hoja < 0.5 la hoja se ve entera dentro del recuadro central de la demo,
    que es el caso en el que su borde podría confundirse con un trazo.
    """
    rng = np.random.default_rng(semilla)
    columnas = np.linspace(0, 1, ancho, dtype=np.float32)[None, :, None]
    mesa = np.array([55, 75, 100], np.float32) * (0.75 + 0.35 * columnas)  # madera con luz lateral (BGR)
    frame = mesa + rng.normal(0, 4, (alto, ancho, 3)).astype(np.float32)

    lado = int(min(ancho, alto) * fraccion_hoja)
    hoja = cv2.resize(foto_simulada(imagen_28, semilla=semilla), (lado, lado), interpolation=cv2.INTER_AREA)
    dx, dy = (float(v) for v in rng.uniform(-0.02, 0.02, size=2) * min(ancho, alto))
    giro = cv2.getRotationMatrix2D((lado / 2, lado / 2), float(rng.uniform(-10, 10)), 1.0)
    giro[0, 2] += (ancho - lado) / 2 + dx
    giro[1, 2] += (alto - lado) / 2 + dy
    hoja = cv2.warpAffine(hoja.astype(np.float32), giro, (ancho, alto))
    cobertura = cv2.warpAffine(np.ones((lado, lado), np.float32), giro, (ancho, alto))[..., None]
    frame = cv2.GaussianBlur(frame * (1 - cobertura) + hoja * cobertura, (3, 3), 0)
    return np.clip(frame, 0, 255).astype(np.uint8)


def dibujo_simulado(imagen_28, lado=280):
    """Trazo blanco sobre negro, como un lienzo de dibujo en la web."""
    grande = cv2.resize(imagen_28.astype(np.float32), (lado, lado), interpolation=cv2.INTER_CUBIC)
    return (np.clip(grande, 0, 1) > 0.4).astype(np.uint8) * 255
