"""Figuras del informe, en versión clara y oscura (el README elige según el tema de GitHub).

Colores: paleta categórica validada (orden fijo, nunca cíclico), una sola escala por
gráfico, rejilla fina y discreta, textos siempre en tinta neutra y etiquetas directas
solo donde aportan. La tabla de resultados del README es la vista accesible de los datos.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

TEMAS = {
    "claro": {
        "superficie": "#fcfcfb",
        "tinta": "#0b0b0b",
        "tinta_2": "#52514e",
        "tenue": "#898781",
        "rejilla": "#e1e0d9",
        "base": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "secuencial": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    },
    "oscuro": {
        "superficie": "#1a1a19",
        "tinta": "#f0efec",
        "tinta_2": "#c3c2b7",
        "tenue": "#898781",
        "rejilla": "#2c2c2a",
        "base": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "secuencial": ["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}

FUENTE = "DejaVu Sans"


def _num(valor, decimales=1):
    """Número con coma decimal y punto de millares (formato español)."""
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def _formato_pct(decimales=0):
    return matplotlib.ticker.FuncFormatter(lambda v, _: f"{_num(v, decimales)} %")


def _estilo(fig, ejes, t):
    fig.patch.set_facecolor(t["superficie"])
    for ax in np.atleast_1d(ejes).ravel():
        ax.set_facecolor(t["superficie"])
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(t["base"])
            ax.spines[lado].set_linewidth(1)
        ax.tick_params(colors=t["tenue"], labelcolor=t["tinta_2"], labelsize=9, length=0)
        ax.grid(True, color=t["rejilla"], linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(t["tinta_2"])
        ax.yaxis.label.set_color(t["tinta_2"])
        ax.title.set_color(t["tinta"])


def _titulo(fig, texto, subtitulo, t):
    """Título y subtítulo a una distancia fija (en pulgadas) del borde superior, sea cual sea el alto."""
    alto = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / alto, texto, ha="left", va="top", fontsize=13, fontweight="bold", color=t["tinta"], family=FUENTE)
    if subtitulo:
        fig.text(0.012, 1 - 0.43 / alto, subtitulo, ha="left", va="top", fontsize=9.5, color=t["tinta_2"], family=FUENTE)


def _guardar(fig, ruta, t):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200, facecolor=t["superficie"])
    plt.close(fig)


def _barra_h(ax, y, ancho, alto, color, superficie, radio):
    """Barra horizontal con el extremo de datos redondeado y el lado de la base recto."""
    if ancho <= 0:
        return
    r = min(radio, ancho / 2)
    caja = FancyBboxPatch(
        (0, y - alto / 2), ancho, alto,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=0, facecolor=color, mutation_aspect=alto / (2 * r) if r else 1,
    )
    ax.add_patch(caja)
    ax.add_patch(Rectangle((0, y - alto / 2), r, alto, linewidth=0, facecolor=color))


def figura_resultados(res, ruta, modo="claro"):
    """Exactitud de cada método en el test original y en el test con giros y desplazamientos."""
    t = TEMAS[modo]
    metodos = res["orden_metodos"]
    nombres = [res["metodos"][m]["nombre"] for m in metodos]
    limpio = [100 * res["metodos"][m]["test"]["exactitud"] for m in metodos]
    perturbado = [100 * res["metodos"][m]["test_perturbado"]["exactitud"] for m in metodos]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.30, right=0.93, top=0.80, bottom=0.12)
    _estilo(fig, ax, t)
    ax.grid(axis="y", visible=False)

    alto, separacion = 0.30, 0.06
    for i, (a, b) in enumerate(zip(limpio, perturbado)):
        y = len(metodos) - 1 - i
        for valor, desplaz, color in ((a, 1, t["series"][0]), (b, -1, t["series"][1])):
            yc = y + desplaz * (alto / 2 + separacion / 2)
            _barra_h(ax, yc, valor, alto, color, t["superficie"], radio=0.9)
            ax.text(valor + 0.8, yc, f"{_num(valor)} %", va="center", ha="left", fontsize=8.5, color=t["tinta_2"])

    ax.set_xlim(0, 108)
    ax.set_ylim(-0.6, len(metodos) - 0.4)
    ax.set_yticks(range(len(metodos)))
    ax.set_yticklabels(nombres[::-1], fontsize=9.5, color=t["tinta"])
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v} %" for v in range(0, 101, 20)])
    ax.set_xlabel("Exactitud en las 10.000 imágenes de test")

    _titulo(
        fig,
        "De la comparación de plantillas a las redes convolucionales",
        "Misma partición para todos los métodos. El test perturbado gira, desplaza y escala cada dígito.",
        t,
    )
    for k, (texto, color) in enumerate(((("Test original"), t["series"][0]), ("Test con giros y desplazamientos", t["series"][1]))):
        x0 = 0.30 + k * 0.20
        fig.patches.append(Rectangle((x0, 0.835), 0.012, 0.022, transform=fig.transFigure, facecolor=color, linewidth=0))
        fig.text(x0 + 0.018, 0.846, texto, va="center", fontsize=9, color=t["tinta_2"])
    _guardar(fig, ruta, t)


def _etiquetas_izquierda(ax, x0, valores, t, separacion=2.6):
    """Coloca las etiquetas a la izquierda del primer punto; si chocan, las separa con una línea guía."""
    orden = np.argsort(valores)
    posiciones = np.array(valores, dtype=float)
    for k in range(1, len(orden)):
        anterior, actual = orden[k - 1], orden[k]
        if posiciones[actual] - posiciones[anterior] < separacion:
            posiciones[actual] = posiciones[anterior] + separacion
    x_texto = x0 / 1.3
    for valor, pos in zip(valores, posiciones):
        if abs(pos - valor) > 0.05:
            ax.plot([x0 / 1.06, x_texto * 1.04], [valor, pos], color=t["tenue"], linewidth=0.8, zorder=2)
        ax.text(x_texto, pos, f"{_num(valor, 1)} %", ha="right", va="center", fontsize=8.5, color=t["tinta_2"])


def figura_curva_aprendizaje(res, ruta, modo="claro"):
    """Exactitud en test según el número de imágenes de entrenamiento (escala logarítmica)."""
    t = TEMAS[modo]
    curva = res["curva_aprendizaje"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13)
    _estilo(fig, ax, t)

    iniciales = []
    for i, clave in enumerate(curva["orden"]):
        serie = curva["series"][clave]
        x, y = curva["tamanos"], [100 * v for v in serie["exactitud_test"]]
        color = t["series"][i]
        ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=serie["nombre"], zorder=3)
        ax.scatter(x, y, s=46, color=color, edgecolors=t["superficie"], linewidths=2, zorder=4)
        iniciales.append(y[0])
    # etiquetas directas en el extremo izquierdo (donde más se separan las curvas), sin solaparse
    _etiquetas_izquierda(ax, curva["tamanos"][0], iniciales, t)

    ax.set_xscale("log")
    ax.set_xlim(curva["tamanos"][0] / 3.0, curva["tamanos"][-1] * 1.35)
    ax.set_xticks(curva["tamanos"])
    ax.set_xticklabels([f"{n:,}".replace(",", ".") for n in curva["tamanos"]])
    ax.minorticks_off()
    ax.set_ylim(min(55, ax.get_ylim()[0]), 100.5)
    ax.yaxis.set_major_formatter(_formato_pct())
    ax.set_xlabel("Imágenes de entrenamiento")
    ax.set_ylabel("Exactitud en test")
    leyenda = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for texto in leyenda.get_texts():
        texto.set_color(t["tinta_2"])
    _titulo(
        fig,
        "¿Cuántos datos necesita cada método?",
        "El test es siempre el mismo y nunca se usa para entrenar ni para ajustar hiperparámetros.",
        t,
    )
    _guardar(fig, ruta, t)


def figura_coste_ssim(res, ruta, modo="claro"):
    """Comparación directa con SSIM: exactitud y tiempo según el número de referencias."""
    t = TEMAS[modo]
    coste = res["coste_ssim"]
    refs = coste["referencias"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.15, wspace=0.28)
    _estilo(fig, (ax1, ax2), t)

    # eje x lineal en los dos paneles: así se ve que la exactitud se satura y el tiempo crece en línea recta
    margen = 0.03 * refs[-1]
    for i, (clave, nombre) in enumerate((("1nn", "Vecino más parecido"), ("media", "Similitud media por dígito"))):
        y = [100 * v for v in coste[clave]]
        ax1.plot(refs, y, color=t["series"][i], linewidth=2, label=nombre, zorder=3)
        ax1.scatter(refs, y, s=46, color=t["series"][i], edgecolors=t["superficie"], linewidths=2, zorder=4)
        ax1.text(refs[-1] + margen, y[-1], f"{_num(y[-1], 1)} %", va="center", fontsize=8.5, color=t["tinta_2"])
    ax1.set_title("Exactitud", loc="left", fontsize=10.5)
    ax1.yaxis.set_major_formatter(_formato_pct())
    leyenda = ax1.legend(loc="center right", frameon=False, fontsize=8.5)
    for texto in leyenda.get_texts():
        texto.set_color(t["tinta_2"])

    ms = coste["ms_por_imagen"]
    ax2.plot(refs, ms, color=t["series"][0], linewidth=2, zorder=3)
    ax2.scatter(refs, ms, s=46, color=t["series"][0], edgecolors=t["superficie"], linewidths=2, zorder=4)
    dec = 0 if max(ms) >= 10 else 1
    ax2.text(refs[-1] + margen, ms[-1], f"{_num(ms[-1], dec)} ms", va="center", fontsize=8.5, color=t["tinta_2"])
    ax2.set_title("Tiempo por imagen (el mismo en los dos métodos)", loc="left", fontsize=10.5)
    ax2.set_ylim(0, max(ms) * 1.15)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{_num(v, dec)} ms"))

    marcas = list(range(0, refs[-1] + 1, 1000))
    for ax in (ax1, ax2):
        ax.set_xticks(marcas)
        ax.set_xticklabels([f"{n:,}".replace(",", ".") for n in marcas])
        ax.set_xlim(0, refs[-1] * 1.2)
        ax.set_xlabel("Imágenes de referencia")
    _titulo(
        fig,
        "Comparar con más ejemplos mejora la exactitud, pero el coste crece en proporción",
        f"Comparación directa con SSIM sobre {coste['n_test']:,} imágenes de test elegidas al azar ".replace(",", ".")
        + "(las mismas en todos los puntos).",
        t,
    )
    _guardar(fig, ruta, t)


def figura_confusion(matriz, titulo, subtitulo, ruta, modo="claro"):
    """Matriz de confusión normalizada por filas (porcentaje de cada dígito real)."""
    t = TEMAS[modo]
    m = np.asarray(matriz, dtype=float)
    porcentaje = 100 * m / m.sum(axis=1, keepdims=True)
    mapa = LinearSegmentedColormap.from_list("secuencial", t["secuencial"])

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.11)
    _estilo(fig, ax, t)
    ax.grid(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    # escala raíz para que los errores pequeños se vean frente a la diagonal
    ax.imshow(np.sqrt(porcentaje / 100), cmap=mapa, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if i == j or porcentaje[i, j] >= 0.5:
                celda_baja = np.sqrt(porcentaje[i, j] / 100) < 0.55
                color_texto = t["tinta"] if celda_baja else t["superficie"]
                ax.text(j, i, _num(porcentaje[i, j]), ha="center", va="center", fontsize=7.5, color=color_texto)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Dígito predicho")
    ax.set_ylabel("Dígito real")
    _titulo(fig, titulo, subtitulo, t)
    _guardar(fig, ruta, t)


def figura_imagenes(imagenes, rotulos, titulo, subtitulo, ruta, modo="claro", columnas=8):
    """Cuadrícula de imágenes en escala de grises con un rótulo bajo cada una."""
    t = TEMAS[modo]
    n = len(imagenes)
    filas = int(np.ceil(n / columnas))
    # alto calculado para que cada celda sea cuadrada: sin huecos de más entre filas
    ancho_fig, hueco_x, hueco_y = 1.25 * columnas, 0.08, 0.30
    lado = 0.98 * ancho_fig / (columnas + (columnas - 1) * hueco_x)
    cabecera, pie = 1.0, 0.05
    alto_fig = cabecera + pie + filas * lado + (filas - 1) * hueco_y * lado
    fig, ejes = plt.subplots(filas, columnas, figsize=(ancho_fig, alto_fig))
    fig.patch.set_facecolor(t["superficie"])
    fig.subplots_adjust(
        left=0.01, right=0.99, top=1 - cabecera / alto_fig, bottom=pie / alto_fig, wspace=hueco_x, hspace=hueco_y
    )
    for k, ax in enumerate(np.atleast_1d(ejes).ravel()):
        ax.set_axis_off()
        if k < n:
            ax.imshow(imagenes[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(rotulos[k], fontsize=8, color=t["tinta_2"], pad=3)
    _titulo(fig, titulo, subtitulo, t)
    _guardar(fig, ruta, t)


def _cuadrada(imagen, margen=0.0):
    """Rellena con negro hasta hacerla cuadrada (y con un margen opcional) para que todos los paneles midan igual."""
    alto, ancho = imagen.shape[:2]
    lado = int(round(max(alto, ancho) * (1 + margen)))
    if lado == alto == ancho:
        return imagen
    salida = np.zeros((lado, lado) + imagen.shape[2:], dtype=imagen.dtype)
    y0, x0 = (lado - alto) // 2, (lado - ancho) // 2
    salida[y0 : y0 + alto, x0 : x0 + ancho] = imagen
    return salida


def figura_pasos(pasos, ruta, modo="claro", subtitulo="El preprocesado reproduce cómo se construyó MNIST."):
    """Pasos del preprocesado de una foto hasta la entrada de la red."""
    t = TEMAS[modo]
    nombres = [
        ("gris", "1. Foto"),
        ("binaria", "2. Umbral adaptativo"),
        ("seleccion", "3. Trazos del dígito"),
        ("recorte", "4. Recorte y grosor"),
        ("entrada_red", "5. Entrada de la red (28x28)"),
    ]
    alto_fig = 3.45
    fig, ejes = plt.subplots(1, 5, figsize=(12, alto_fig))
    fig.patch.set_facecolor(t["superficie"])
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.0 / alto_fig, bottom=0.08 / alto_fig, wspace=0.12)
    for ax, (clave, nombre) in zip(ejes, nombres):
        ax.set_axis_off()
        imagen = pasos[clave]
        if imagen.ndim == 3:  # foto en color de OpenCV (BGR)
            ax.imshow(_cuadrada(imagen)[..., ::-1])
        else:
            if clave == "gris":
                nombre = "1. Escala de grises"
            ax.imshow(_cuadrada(imagen, margen=0.15 if clave == "recorte" else 0.0), cmap="gray")
        ax.set_title(nombre, fontsize=9, color=t["tinta_2"], pad=6)
    _titulo(fig, "De una foto a la entrada de la red", subtitulo, t)
    _guardar(fig, ruta, t)


def figura_entrenamiento(res, ruta, modo="claro"):
    """Exactitud en validación tras cada epoch para las tres redes."""
    t = TEMAS[modo]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.78, bottom=0.14)
    _estilo(fig, ax, t)
    ultima, maximo = 1, 0.0
    for i, clave in enumerate(("mlp", "lenet", "cnn_aumentada")):
        info = res["metodos"][clave]["entrenamiento"]
        epocas = [p / info["pasos_por_epoca"] for p in info["historial"]["paso"]]
        y = [100 * v for v in info["historial"]["exactitud_val"]]
        color = t["series"][i + 1]
        ax.plot(epocas, y, color=color, linewidth=2, label=res["metodos"][clave]["nombre"], zorder=3)
        ax.scatter(epocas, y, s=30, color=color, edgecolors=t["superficie"], linewidths=1.5, zorder=4)
        # anillo en el epoch que se guarda, con su exactitud
        mejor_epoca = info["mejor_paso"] / info["pasos_por_epoca"]
        mejor = 100 * info["mejor_exactitud_val"]
        ax.scatter([mejor_epoca], [mejor], s=120, facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
        ax.annotate(
            f"{_num(mejor, 2)} %", (mejor_epoca, mejor), xytext=(0, 9), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color=t["tinta_2"],
        )
        ultima, maximo = max(ultima, max(epocas)), max(maximo, max(y))
    ax.set_xticks([1] + list(range(5, int(round(ultima)) + 1, 5)))
    ax.set_xlim(0.3, ultima + 0.7)
    ax.set_ylim(top=maximo + 0.35)
    ax.yaxis.set_major_formatter(_formato_pct(1))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Exactitud en validación")
    leyenda = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for texto in leyenda.get_texts():
        texto.set_color(t["tinta_2"])
    _titulo(
        fig, "Entrenamiento de las redes",
        "El anillo marca el epoch que se guarda: el de mejor exactitud en validación (parada temprana).", t,
    )
    _guardar(fig, ruta, t)


def generar_todas(res, carpeta, extras=None):
    """Genera todas las figuras en las dos versiones (clara y oscura)."""
    carpeta = Path(carpeta)
    for modo in TEMAS:
        sufijo = "" if modo == "claro" else "_oscuro"
        figura_resultados(res, carpeta / f"resultados{sufijo}.png", modo)
        figura_curva_aprendizaje(res, carpeta / f"curva_aprendizaje{sufijo}.png", modo)
        figura_coste_ssim(res, carpeta / f"coste_ssim{sufijo}.png", modo)
        figura_entrenamiento(res, carpeta / f"entrenamiento{sufijo}.png", modo)
        mejor = res["metodos"]["cnn_aumentada"]
        figura_confusion(
            mejor["test"]["matriz_confusion"],
            "Dónde se equivoca la CNN",
            "Porcentaje de cada dígito real (filas). Se rotulan la diagonal y los errores de 0,5 % o más.",
            carpeta / f"confusion_cnn{sufijo}.png",
            modo,
        )
        plantillas = res["metodos"]["plantillas_ssim"]
        figura_confusion(
            plantillas["test"]["matriz_confusion"],
            "Dónde se equivocan las plantillas",
            "Porcentaje de cada dígito real (filas). Se rotulan la diagonal y los errores de 0,5 % o más.",
            carpeta / f"confusion_plantillas{sufijo}.png",
            modo,
        )
        if extras:
            figura_imagenes(
                extras["plantillas"], [f"plantilla {d}" for d in range(10)],
                "Las diez plantillas: la imagen media de cada dígito",
                "Una imagen nueva se asigna a la plantilla con mayor SSIM.",
                carpeta / f"plantillas{sufijo}.png", modo, columnas=10,
            )
            figura_imagenes(
                extras["errores"], extras["rotulos_errores"],
                "Los fallos más confiados de la CNN",
                "Real → predicho (confianza). Muchos son ambiguos incluso para una persona.",
                carpeta / f"errores_cnn{sufijo}.png", modo,
            )
            if "pasos" in extras:
                figura_pasos(
                    extras["pasos"], carpeta / f"preprocesado{sufijo}.png", modo,
                    subtitulo=extras.get("subtitulo_pasos", "El preprocesado reproduce cómo se construyó MNIST."),
                )


def extras_desde_modelo():
    """Recalcula las figuras de imágenes (plantillas, fallos y preprocesado) sin reentrenar nada."""
    import torch

    from .datos import cargar_mnist
    from .evaluacion import errores_mas_confiados
    from .modelos import crear_modelo, predecir_probabilidades
    from .preprocesado import preprocesar
    from .simulacion import foto_simulada

    raiz = Path(__file__).resolve().parent.parent
    d = cargar_mnist()
    modelo = crear_modelo("cnn_aumentada")
    modelo.load_state_dict(torch.load(raiz / "models" / "cnn_aumentada.pt", map_location="cpu"))
    prob = predecir_probabilidades(modelo, d["X_test"])
    pred = prob.argmax(1)
    indices = errores_mas_confiados(prob, d["y_test"], n=16)
    extras = {
        "plantillas": [d["X_train"][d["y_train"] == c].mean(axis=0) for c in range(10)],
        "errores": [d["X_test"][i] for i in indices],
        "rotulos_errores": [f"{d['y_test'][i]} → {pred[i]} ({confianza(prob[i, pred[i]])})" for i in indices],
    }
    # foto simulada (papel, sombra, tinta, ruido y giro) de un dígito del test; con una foto real:
    # python -m src.figura_preprocesado foto.jpg
    foto = foto_simulada(d["X_test"][0])
    resultado = preprocesar(foto)
    if resultado.imagen_28 is not None:
        p = predecir_probabilidades(modelo, resultado.imagen_28[None])[0]
        extras["pasos"] = {**resultado.pasos, "gris": foto}
        extras["subtitulo_pasos"] = (
            f"Foto simulada de un dígito escrito con bolígrafo. La red reconoce un {int(p.argmax())} "
            f"{frase_confianza(p.max())}."
        )
    return extras


def confianza(p):
    """Probabilidad como porcentaje con un decimal; nunca muestra un «100 %» redondeado."""
    return "> 99,9 %" if p >= 0.9995 else f"{_num(100 * p, 1)} %"


def frase_confianza(p):
    """«con un 97,3 % de confianza» o «con más de un 99,9 % de confianza»."""
    return "con más de un 99,9 % de confianza" if p >= 0.9995 else f"con un {_num(100 * p, 1)} % de confianza"


if __name__ == "__main__":
    # python -m src.graficas  -> regenera todas las figuras a partir de reports/metricas.json y del modelo guardado
    raiz = Path(__file__).resolve().parent.parent
    resultados = json.loads((raiz / "reports" / "metricas.json").read_text(encoding="utf-8"))
    generar_todas(resultados, raiz / "reports" / "figuras", extras_desde_modelo())
