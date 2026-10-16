"""Figuras del README, a partir de reports/metricas.json y del modelo guardado.

Para regenerarlas sin volver a entrenar:

    python -m src.graficas

Todas siguen el mismo estilo: fondo blanco, rejilla suave, los mismos colores en el
mismo orden y números en formato español (coma decimal).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # dibuja directamente en archivos, sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "reports" / "figuras"

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # azul, naranja, verde y amarillo
TEXTO_SUAVE = "#52514e"  # gris oscuro para ejes, leyendas y valores

# Estilo común de todas las figuras
plt.rcParams.update(
    {
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.spines.top": False,  # sin bordes arriba ni a la derecha
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.grid": True,
        "axes.axisbelow": True,  # la rejilla, por detrás de los datos
        "grid.color": "#e1e0d9",
        "axes.labelcolor": TEXTO_SUAVE,
        "xtick.color": TEXTO_SUAVE,
        "ytick.color": TEXTO_SUAVE,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "font.size": 9.5,
        "legend.frameon": False,
    }
)


# --- Formato de los números -------------------------------------------------------------


def numero(valor, decimales=1):
    """Número en formato español: 1234.5 -> «1.234,5»."""
    texto = f"{valor:,.{decimales}f}"  # formato inglés: 1,234.5
    return texto.replace(",", " ").replace(".", ",").replace(" ", ".")


def porcentaje(fraccion, decimales=1):
    """0.9943 -> «99,4 %»."""
    return f"{numero(100 * fraccion, decimales)} %"


def confianza(p):
    """Probabilidad como porcentaje; nunca muestra un «100 %» redondeado."""
    return "> 99,9 %" if p >= 0.9995 else porcentaje(p)


def frase_confianza(p):
    """«con un 97,3 % de confianza» o «con más de un 99,9 % de confianza»."""
    return "con más de un 99,9 % de confianza" if p >= 0.9995 else f"con un {porcentaje(p)} de confianza"


# --- Piezas comunes ---------------------------------------------------------------------


def cabecera(fig, titulo, subtitulo=""):
    """Título en negrita y subtítulo en gris, arriba a la izquierda de la figura."""
    alto = fig.get_figheight()  # en pulgadas: así el título queda siempre a la misma distancia del borde
    fig.text(0.012, 1 - 0.12 / alto, titulo, ha="left", va="top", fontsize=13, fontweight="bold")
    if subtitulo:
        fig.text(0.012, 1 - 0.43 / alto, subtitulo, ha="left", va="top", fontsize=9.5, color=TEXTO_SUAVE)


def guardar(fig, ruta):
    """Guarda la figura en PNG y libera la memoria."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta)
    plt.close(fig)


# --- Gráficos de resultados -------------------------------------------------------------


def figura_resultados(res, ruta):
    """Barras con la exactitud de cada método en el test original y en el perturbado."""
    metodos = res["orden_metodos"][::-1]  # al revés: el método más sencillo queda arriba y la mejor red, abajo
    nombres = [res["metodos"][m]["nombre"] for m in metodos]
    original = [res["metodos"][m]["test"]["exactitud"] for m in metodos]
    perturbado = [res["metodos"][m]["test_perturbado"]["exactitud"] for m in metodos]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.11)
    y = np.arange(len(metodos))
    series = [
        (0.18, original, COLORES[0], "Test original"),
        (-0.18, perturbado, COLORES[1], "Test con giros y desplazamientos"),
    ]
    for desplazamiento, valores, color, etiqueta in series:
        ax.barh(y + desplazamiento, valores, height=0.34, color=color, label=etiqueta)
        for yi, valor in zip(y + desplazamiento, valores):  # el valor al final de cada barra
            ax.text(valor + 0.008, yi, porcentaje(valor), va="center", fontsize=8.5, color=TEXTO_SUAVE)

    ax.set_yticks(y, nombres, fontsize=9.5, color="black")
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v} %" for v in range(0, 101, 20)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Exactitud en las 10.000 imágenes de test")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)  # leyenda encima del gráfico
    cabecera(
        fig,
        "De la comparación de plantillas a las redes convolucionales",
        "Misma partición para todos los métodos. El test perturbado gira, desplaza y escala cada dígito.",
    )
    guardar(fig, ruta)


def figura_curva_aprendizaje(res, ruta):
    """Exactitud en test según el número de imágenes de entrenamiento."""
    curva = res["curva_aprendizaje"]
    tamanos = curva["tamanos"]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
    for color, clave in zip(COLORES, curva["orden"]):
        serie = curva["series"][clave]
        ax.plot(tamanos, serie["exactitud_test"], color=color, linewidth=2, marker="o",
                markersize=6, markeredgecolor="white", label=serie["nombre"])

    ax.set_xscale("log")  # escala logarítmica: de 100 a 55.000 imágenes
    ax.set_xticks(tamanos, [numero(n, 0) for n in tamanos])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(lambda v, _: porcentaje(v, 0))
    ax.set_xlabel("Imágenes de entrenamiento (escala logarítmica)")
    ax.set_ylabel("Exactitud en test")
    ax.legend(loc="lower right")
    cabecera(
        fig,
        "¿Cuántos datos necesita cada método?",
        "El test es siempre el mismo y nunca se usa para entrenar ni para ajustar hiperparámetros.",
    )
    guardar(fig, ruta)


def figura_coste_ssim(res, ruta):
    """Comparación directa con SSIM: exactitud (izquierda) y tiempo (derecha) según las referencias."""
    coste = res["coste_ssim"]
    referencias = coste["referencias"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.15, wspace=0.28)

    for color, clave, nombre in ((COLORES[0], "1nn", "Vecino más parecido"), (COLORES[1], "media", "Similitud media por dígito")):
        ax1.plot(referencias, coste[clave], color=color, linewidth=2, marker="o", markeredgecolor="white", label=nombre)
    ax1.set_title("Exactitud", loc="left", fontsize=10.5)
    ax1.yaxis.set_major_formatter(lambda v, _: porcentaje(v, 0))
    ax1.legend(loc="center right", fontsize=8.5)

    ax2.plot(referencias, coste["ms_por_imagen"], color=COLORES[0], linewidth=2, marker="o", markeredgecolor="white")
    ax2.set_title("Tiempo por imagen (el mismo en los dos métodos)", loc="left", fontsize=10.5)
    ax2.set_ylim(0, max(coste["ms_por_imagen"]) * 1.15)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{numero(v, 0)} ms")

    for ax in (ax1, ax2):
        ax.set_xlabel("Imágenes de referencia")
        ax.xaxis.set_major_formatter(lambda v, _: numero(v, 0))
    cabecera(
        fig,
        "Comparar con más ejemplos mejora la exactitud, pero el coste crece en proporción",
        f"Comparación directa con SSIM sobre {numero(coste['n_test'], 0)} imágenes de test elegidas al azar "
        "(las mismas en todos los puntos).",
    )
    guardar(fig, ruta)


def figura_entrenamiento(res, ruta):
    """Exactitud en validación tras cada epoch para las tres redes."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    for color, clave in zip(COLORES[1:], ("mlp", "lenet", "cnn_aumentada")):
        info = res["metodos"][clave]["entrenamiento"]
        epocas = [paso / info["pasos_por_epoca"] for paso in info["historial"]["paso"]]
        ax.plot(epocas, info["historial"]["exactitud_val"], color=color, linewidth=2, marker="o",
                markersize=5, markeredgecolor="white", label=res["metodos"][clave]["nombre"])
        # Anillo en el epoch que se guarda (el de mejor validación) con su exactitud
        mejor_epoca = info["mejor_paso"] / info["pasos_por_epoca"]
        mejor = info["mejor_exactitud_val"]
        ax.scatter([mejor_epoca], [mejor], s=120, facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
        ax.annotate(porcentaje(mejor, 2), (mejor_epoca, mejor), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=TEXTO_SUAVE)

    ax.set_xticks([1, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter(lambda v, _: porcentaje(v, 1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Exactitud en validación")
    ax.legend(loc="lower right")
    cabecera(
        fig,
        "Entrenamiento de las redes",
        "El anillo marca el epoch que se guarda: el de mejor exactitud en validación (parada temprana).",
    )
    guardar(fig, ruta)


def figura_confusion(matriz, titulo, ruta):
    """Matriz de confusión: porcentaje de cada dígito real (filas) que se predice como cada dígito."""
    m = np.asarray(matriz, dtype=float)
    porcentajes = m / m.sum(axis=1, keepdims=True)  # cada fila suma 1

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.09)
    ax.grid(False)
    # Raíz cuadrada en el color para que los errores pequeños se vean al lado de la diagonal
    ax.imshow(np.sqrt(porcentajes), cmap="Blues", vmin=0, vmax=1)
    for real in range(10):
        for predicho in range(10):
            valor = porcentajes[real, predicho]
            if real == predicho or valor >= 0.005:  # se rotulan la diagonal y los errores de 0,5 % o más
                color = "white" if np.sqrt(valor) > 0.6 else "black"
                ax.text(predicho, real, numero(100 * valor), ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Dígito predicho")
    ax.set_ylabel("Dígito real")
    cabecera(fig, titulo, "Porcentaje de cada dígito real (filas). Se rotulan la diagonal y los errores de 0,5 % o más.")
    guardar(fig, ruta)


# --- Figuras con imágenes ---------------------------------------------------------------


def figura_imagenes(imagenes, rotulos, titulo, subtitulo, ruta, columnas=8):
    """Cuadrícula de imágenes en escala de grises con un rótulo encima de cada una."""
    filas = int(np.ceil(len(imagenes) / columnas))
    ancho = 1.25 * columnas
    alto = 1.0 + 1.45 * filas  # 1 pulgada para la cabecera y unas 1,45 por fila
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho, alto))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / alto, bottom=0.02, wspace=0.1, hspace=0.35)
    for k, ax in enumerate(np.atleast_1d(ejes).ravel()):
        ax.set_axis_off()
        if k < len(imagenes):
            ax.imshow(imagenes[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(rotulos[k], fontsize=8, color=TEXTO_SUAVE, pad=3)
    cabecera(fig, titulo, subtitulo)
    guardar(fig, ruta)


def cuadrada(imagen, margen=0.0):
    """Rellena la imagen con negro hasta hacerla cuadrada, para que todos los paneles midan igual."""
    alto, ancho = imagen.shape[:2]
    lado = int(round(max(alto, ancho) * (1 + margen)))
    salida = np.zeros((lado, lado) + imagen.shape[2:], dtype=imagen.dtype)
    y0, x0 = (lado - alto) // 2, (lado - ancho) // 2
    salida[y0 : y0 + alto, x0 : x0 + ancho] = imagen
    return salida


def figura_pasos(pasos, ruta, subtitulo):
    """Los cinco pasos del preprocesado de una foto hasta la entrada de la red."""
    paneles = [
        ("gris", "1. Foto"),
        ("binaria", "2. Umbral adaptativo"),
        ("seleccion", "3. Trazos del dígito"),
        ("recorte", "4. Recorte y grosor"),
        ("entrada_red", "5. Entrada de la red (28x28)"),
    ]
    alto = 3.45
    fig, ejes = plt.subplots(1, 5, figsize=(12, alto))
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / alto, bottom=0.03, wspace=0.12)
    for ax, (clave, nombre) in zip(ejes, paneles):
        ax.set_axis_off()
        imagen = pasos[clave]
        if imagen.ndim == 3:  # foto en color de OpenCV: se pasa de BGR a RGB para dibujarla
            ax.imshow(cuadrada(imagen)[..., ::-1])
        else:
            ax.imshow(cuadrada(imagen, margen=0.15 if clave == "recorte" else 0.0), cmap="gray")
        ax.set_title(nombre, fontsize=9, color=TEXTO_SUAVE, pad=6)
    cabecera(fig, "De una foto a la entrada de la red", subtitulo)
    guardar(fig, ruta)


# --- Todas las figuras ------------------------------------------------------------------


def generar_todas(res, extras, carpeta=FIGURAS):
    """Genera todas las figuras del README en la carpeta indicada."""
    carpeta = Path(carpeta)
    figura_resultados(res, carpeta / "resultados.png")
    figura_curva_aprendizaje(res, carpeta / "curva_aprendizaje.png")
    figura_coste_ssim(res, carpeta / "coste_ssim.png")
    figura_entrenamiento(res, carpeta / "entrenamiento.png")
    figura_confusion(res["metodos"]["cnn_aumentada"]["test"]["matriz_confusion"], "Dónde se equivoca la CNN",
                     carpeta / "confusion_cnn.png")
    figura_confusion(res["metodos"]["plantillas_ssim"]["test"]["matriz_confusion"], "Dónde se equivocan las plantillas",
                     carpeta / "confusion_plantillas.png")
    figura_imagenes(extras["plantillas"], [f"plantilla {d}" for d in range(10)],
                    "Las diez plantillas: la imagen media de cada dígito",
                    "Una imagen nueva se asigna a la plantilla con mayor SSIM.",
                    carpeta / "plantillas.png", columnas=10)
    figura_imagenes(extras["errores"], extras["rotulos_errores"],
                    "Los fallos más confiados de la CNN",
                    "Real → predicho (confianza). Muchos son ambiguos incluso para una persona.",
                    carpeta / "errores_cnn.png")
    figura_pasos(extras["pasos"], carpeta / "preprocesado.png", extras["subtitulo_pasos"])


def extras_desde_modelo():
    """Datos de las figuras con imágenes (plantillas, fallos y preprocesado), sin reentrenar.

    Usa MNIST y la red guardada en models/cnn_aumentada.pt.
    """
    import torch

    from .datos import cargar_mnist
    from .evaluacion import errores_mas_confiados
    from .modelos import crear_modelo, predecir_probabilidades
    from .preprocesado import preprocesar
    from .simulacion import foto_simulada

    d = cargar_mnist()
    modelo = crear_modelo("cnn_aumentada")
    modelo.load_state_dict(torch.load(RAIZ / "models" / "cnn_aumentada.pt", map_location="cpu"))

    # Los 16 fallos del test en los que la red estaba más segura
    prob = predecir_probabilidades(modelo, d["X_test"])
    pred = prob.argmax(1)
    indices = errores_mas_confiados(prob, d["y_test"], n=16)

    # Foto simulada (papel, sombra, tinta, ruido y giro) del primer dígito del test.
    # Para usar una foto real: python -m src.figura_preprocesado foto.jpg
    foto = foto_simulada(d["X_test"][0])
    resultado = preprocesar(foto)
    p = predecir_probabilidades(modelo, resultado.imagen_28[None])[0]

    return {
        "plantillas": [d["X_train"][d["y_train"] == c].mean(axis=0) for c in range(10)],
        "errores": [d["X_test"][i] for i in indices],
        "rotulos_errores": [f"{d['y_test'][i]} → {pred[i]} ({confianza(prob[i, pred[i]])})" for i in indices],
        "pasos": {**resultado.pasos, "gris": foto},
        "subtitulo_pasos": (
            f"Foto simulada de un dígito escrito con bolígrafo. La red reconoce un {int(p.argmax())} "
            f"{frase_confianza(p.max())}."
        ),
    }


if __name__ == "__main__":
    resultados = json.loads((RAIZ / "reports" / "metricas.json").read_text(encoding="utf-8"))
    generar_todas(resultados, extras_desde_modelo())
    print(f"Figuras guardadas en {FIGURAS}")
