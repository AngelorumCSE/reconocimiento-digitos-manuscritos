"""Web de demostración (Streamlit): reconoce un dígito escrito a mano desde la cámara o una foto.

Ejecución en local:
    streamlit run app_digitos.py

Publicada en https://digitos-mnist.streamlit.app (Streamlit Community Cloud la
despliega desde este repositorio e instala solo requirements.txt).
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.inferencia import Clasificador
from src.preprocesado import preprocesar

RAIZ = Path(__file__).resolve().parent
FIGURAS = RAIZ / "reports" / "figuras"
TABLA = RAIZ / "reports" / "resultados.md"
UMBRAL_CONFIANZA = 0.6  # por debajo de esta confianza, la web responde «?»

st.set_page_config(page_title="Reconocimiento de dígitos manuscritos", layout="wide")


@st.cache_resource  # el modelo se carga una sola vez, no en cada interacción
def cargar_clasificador():
    """La red exportada a ONNX, lista para clasificar."""
    return Clasificador()


def leer_imagen(archivo):
    """Convierte el archivo subido (o la foto de la cámara) en una imagen de OpenCV (BGR)."""
    datos = np.frombuffer(archivo.getvalue(), np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def porcentaje(p):
    """Probabilidad con coma decimal; nunca muestra un «100 %» redondeado."""
    return "más del 99,9 %" if p >= 0.9995 else f"{100 * p:.1f} %".replace(".", ",")


def ampliar(imagen, lado=196):
    """Amplía una imagen pequeña (como la de 28x28) para verla bien, sin suavizar los píxeles."""
    imagen = (imagen * 255).astype(np.uint8) if imagen.dtype != np.uint8 else imagen
    return cv2.resize(imagen, (lado, lado), interpolation=cv2.INTER_NEAREST)


def mostrar_prediccion(imagen_bgr, fondo="auto"):
    """Preprocesa la imagen, la clasifica y enseña cada paso y la probabilidad de cada dígito."""
    resultado = preprocesar(imagen_bgr, fondo=fondo)
    if resultado.imagen_28 is None:
        st.warning(
            "No encuentro ningún dígito. Prueba con el dígito más grande y centrado, un trazo más "
            "grueso (rotulador) y buena luz, sin otros objetos en la imagen."
        )
        return

    prob = cargar_clasificador().probabilidades(resultado.imagen_28)
    mejor = int(np.argmax(prob))
    confianza = float(prob[mejor])

    # Recuadro azul alrededor del dígito encontrado
    original = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    x, y, w, h = resultado.caja
    grosor = max(2, min(original.shape[:2]) // 150)
    cv2.rectangle(original, (x, y), (x + w, y + h), (42, 120, 214), grosor)

    col1, col2, col3 = st.columns([1.3, 1, 1.2])
    with col1:
        st.image(original, caption="Imagen y dígito detectado", width="stretch")
    with col2:
        st.image(ampliar(resultado.binaria), caption="Umbral adaptativo", width="stretch")
        st.image(ampliar(resultado.imagen_28), caption="Entrada de la red (28x28, estilo MNIST)", width="stretch")
    with col3:
        if confianza >= UMBRAL_CONFIANZA:
            st.metric("Dígito reconocido", str(mejor))
            st.caption(f"Confianza: {porcentaje(confianza)}")
        else:
            st.metric("Dígito reconocido", "?")
            st.caption(
                f"Confianza: solo {porcentaje(confianza)}. La red duda: puede que el dígito sea ambiguo "
                "o que el recorte no sea bueno."
            )
        tabla = pd.DataFrame({"Probabilidad (%)": 100 * prob}, index=[str(d) for d in range(10)])
        st.bar_chart(tabla, horizontal=True, sort=False, color="#2a78d6", height=320)


# --- La página --------------------------------------------------------------------------

st.title("Reconocimiento de dígitos manuscritos")
st.write(
    "Escribe un dígito (0-9) en un papel y muéstralo a la cámara o sube una foto. La imagen se "
    "preprocesa para que se parezca a las de MNIST y la clasifica una red convolucional entrenada "
    "con aumento de datos en PyTorch."
)

pestana_camara, pestana_foto, pestana_estudio = st.tabs(["Cámara", "Subir foto", "Comparativa de métodos"])

with pestana_camara:
    captura = st.camera_input("Haz una foto del dígito")
    st.caption("Acerca el papel a la cámara para que el dígito salga grande y centrado.")
    if captura is not None:
        mostrar_prediccion(leer_imagen(captura), fondo="claro")  # con la cámara siempre es tinta sobre papel

with pestana_foto:
    archivo = st.file_uploader("Foto de un dígito escrito a mano", type=["png", "jpg", "jpeg"])
    if archivo is not None:
        mostrar_prediccion(leer_imagen(archivo))

with pestana_estudio:
    st.write(
        "Siete métodos evaluados con la misma partición de MNIST: del más sencillo (comparar con "
        "plantillas mediante SSIM) a una red convolucional con aumento de datos. El test perturbado "
        "gira, desplaza y escala cada dígito para medir la robustez."
    )
    if TABLA.exists():
        st.markdown(TABLA.read_text(encoding="utf-8"))
    for nombre, pie in [
        ("resultados.png", "Exactitud de cada método en el test original y en el perturbado."),
        ("curva_aprendizaje.png", "Exactitud según el número de imágenes de entrenamiento."),
        ("confusion_cnn.png", "Errores de la red convolucional por dígito."),
    ]:
        ruta = FIGURAS / nombre
        if ruta.exists():
            st.image(str(ruta), caption=pie, width="stretch")

st.caption("La red se ejecuta con ONNX Runtime. Código y metodología en el README del repositorio.")
