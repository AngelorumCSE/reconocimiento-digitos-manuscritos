"""Reproduce todos los experimentos y guarda las métricas, el modelo y las figuras.

Uso (desde la carpeta del proyecto):

    python -m src.experimentos

Tarda algo más de media hora en un portátil con 2 núcleos y sin GPU. Hace, en orden:

1. Comparación directa con SSIM (vecino más parecido y similitud media por dígito).
2. Plantillas medias + SSIM.
3. HOG + SVM lineal.
4. Tres redes neuronales: perceptrón, LeNet y LeNet mejorada con aumento de datos.
5. Exporta la mejor red a ONNX y mide la cadena completa con fotos simuladas.
6. Curva de aprendizaje: cada método entrenado con 100 a 55.000 imágenes.
7. Guarda reports/metricas.json, reports/resultados.md y las figuras de reports/figuras/.

El orden importa: con las mismas semillas, repetir los pasos en el mismo orden da
exactamente los mismos resultados.
"""

import json
import platform
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning

from .datos import SEMILLA, cargar_mnist, perturbar, subconjunto_estratificado
from .entrenamiento import entrenar
from .evaluacion import latencia_ms, metricas
from .graficas import extras_desde_modelo, generar_todas
from .metodos_clasicos import PlantillasSSIM, VecinoSSIM, caracteristicas_hog, crear_svm_hog
from .modelos import crear_modelo, n_parametros, predecir_probabilidades
from .preprocesado import preprocesar, recuadro_central
from .simulacion import foto_simulada, fotograma_simulado

RAIZ = Path(__file__).resolve().parent.parent
REPORTS = RAIZ / "reports"
MODELOS = RAIZ / "models"

N_REFERENCIAS = 1000  # referencias (100 por dígito) para la comparación directa con SSIM
REFERENCIAS_COSTE = [100, 300, 1000, 3000]  # puntos de la figura de coste
N_TEST_COSTE = 1000
ALPHAS_SVM = [1e-5, 1e-4, 1e-3]  # regularizaciones del SVM que se prueban
TAMANOS_CURVA = [100, 300, 1000, 3000, 10000, 30000, 55000]
N_SIMULADAS = 1000  # fotos y fotogramas simulados para medir la cadena completa (100 por dígito)

NOMBRES = {
    "media_ssim": "Similitud media por dígito (SSIM)",
    "plantillas_ssim": "Plantillas medias + SSIM",
    "vecino_ssim": "Vecino más parecido (SSIM)",
    "hog_svm": "HOG + SVM lineal",
    "mlp": "Perceptrón multicapa (PyTorch)",
    "lenet": "CNN LeNet (PyTorch)",
    "cnn_aumentada": "CNN + aumento de datos",
}
CONFIG_REDES = {
    "mlp": {"epocas": 15, "aumentar": False},
    "lenet": {"epocas": 15, "aumentar": False},
    "cnn_aumentada": {"epocas": 20, "aumentar": True, "paciencia": 4},
}


def registro(nombre, y_test, pred, pred_perturbado, segundos, ms, parametros=None, extra=None):
    """Resultados de un método en el formato de metricas.json."""
    salida = {
        "nombre": nombre,
        "test": metricas(y_test, pred),
        "test_perturbado": metricas(y_test, pred_perturbado),
        "segundos_entrenamiento": float(segundos),
        "ms_por_imagen": float(ms),
        "parametros": parametros,
    }
    if extra:
        salida.update(extra)
    return salida


def imprimir(clave, r):
    """Resumen de una línea de los resultados de un método."""
    print(
        f"[{clave}] test={r['test']['exactitud']:.4f}  perturbado={r['test_perturbado']['exactitud']:.4f}  "
        f"entreno={r['segundos_entrenamiento']:.1f}s  {r['ms_por_imagen']:.2f} ms/img",
        flush=True,
    )


# --- 1. Comparación directa con SSIM ----------------------------------------------------


def experimento_vecino_ssim(d, X_test_p, res):
    """Compara cada imagen del test con 1.000 referencias del entrenamiento."""
    X_ref, y_ref = subconjunto_estratificado(d["X_train"], d["y_train"], N_REFERENCIAS)
    vecino = VecinoSSIM().fit(X_ref, y_ref)
    inicio = time.perf_counter()
    S = vecino.similitudes(d["X_test"])  # se calcula una vez y sirve para los dos modos
    S_p = vecino.similitudes(X_test_p)
    segundos = time.perf_counter() - inicio
    ms = latencia_ms(lambda x: vecino.predict(x), d["X_test"], n=50)
    for clave, modo in (("media_ssim", "media"), ("vecino_ssim", "1nn")):
        res["metodos"][clave] = registro(
            NOMBRES[clave], d["y_test"], vecino.decidir(S, y_ref, modo), vecino.decidir(S_p, y_ref, modo),
            0.0, ms, extra={"referencias": N_REFERENCIAS, "segundos_test": segundos / 2},
        )
        imprimir(clave, res["metodos"][clave])


def medir_coste_ssim(d):
    """Exactitud y tiempo de la comparación directa según el número de referencias.

    Se usan las mismas 1.000 imágenes del test (100 por dígito) en todos los puntos.
    """
    X_tc, y_tc = subconjunto_estratificado(d["X_test"], d["y_test"], N_TEST_COSTE)
    coste = {"referencias": REFERENCIAS_COSTE, "n_test": N_TEST_COSTE, "1nn": [], "media": [], "ms_por_imagen": []}
    for n_ref in REFERENCIAS_COSTE:
        Xr, yr = subconjunto_estratificado(d["X_train"], d["y_train"], n_ref)
        vecino = VecinoSSIM().fit(Xr, yr)
        inicio = time.perf_counter()
        S = vecino.similitudes(X_tc)
        coste["ms_por_imagen"].append((time.perf_counter() - inicio) / N_TEST_COSTE * 1000)
        for modo in ("1nn", "media"):
            coste[modo].append(float((vecino.decidir(S, yr, modo) == y_tc).mean()))
        print(f"  coste SSIM refs={n_ref}: 1nn={coste['1nn'][-1]:.3f} media={coste['media'][-1]:.3f}", flush=True)
    return coste


# --- 2. Plantillas medias + SSIM --------------------------------------------------------


def experimento_plantillas(d, X_test_p, res):
    """Una plantilla media por dígito; cada imagen va a la plantilla con mayor SSIM."""
    inicio = time.perf_counter()
    plantillas = PlantillasSSIM().fit(d["X_train"], d["y_train"])
    segundos = time.perf_counter() - inicio
    res["metodos"]["plantillas_ssim"] = registro(
        NOMBRES["plantillas_ssim"], d["y_test"], plantillas.predict(d["X_test"]), plantillas.predict(X_test_p),
        segundos, latencia_ms(lambda x: plantillas.predict(x), d["X_test"]),
    )
    imprimir("plantillas_ssim", res["metodos"]["plantillas_ssim"])


# --- 3. HOG + SVM lineal ----------------------------------------------------------------


def ajustar_svm(F_train, y_train, F_val, y_val):
    """Entrena un SVM con cada regularización y se queda con el mejor en validación.

    Devuelve (exactitud en validación, alpha elegido, modelo, segundos de ese entrenamiento).
    """
    mejor = None
    for alpha in ALPHAS_SVM:
        inicio = time.perf_counter()
        with warnings.catch_warnings():
            # con 30 pasadas, SGD a veces no llega a la tolerancia, pero la validación ya es estable
            warnings.simplefilter("ignore", ConvergenceWarning)
            modelo = crear_svm_hog(alpha).fit(F_train, y_train)
        segundos = time.perf_counter() - inicio
        acc = float((modelo.predict(F_val) == y_val).mean())
        if mejor is None or acc > mejor[0]:
            mejor = (acc, alpha, modelo, segundos)
    return mejor


def experimento_hog_svm(d, X_test_p, res):
    """Devuelve las características HOG, que se reutilizan en la curva de aprendizaje."""
    inicio = time.perf_counter()
    hog = {"train": caracteristicas_hog(d["X_train"])}
    segundos_hog = time.perf_counter() - inicio
    hog["val"], hog["test"], hog["test_p"] = (caracteristicas_hog(X) for X in (d["X_val"], d["X_test"], X_test_p))

    acc_val, alpha, svm, segundos_svm = ajustar_svm(hog["train"], d["y_train"], hog["val"], d["y_val"])
    ms = latencia_ms(lambda x: svm.predict(caracteristicas_hog(x, n_jobs=1)), d["X_test"])
    res["metodos"]["hog_svm"] = registro(
        NOMBRES["hog_svm"], d["y_test"], svm.predict(hog["test"]), svm.predict(hog["test_p"]),
        segundos_hog + segundos_svm, ms,
        parametros=int(svm[-1].coef_.size + svm[-1].intercept_.size),
        extra={"alpha": alpha, "exactitud_val": acc_val, "dimension_hog": int(hog["train"].shape[1])},
    )
    imprimir("hog_svm", res["metodos"]["hog_svm"])
    return hog


# --- 4. Redes neuronales ----------------------------------------------------------------


def experimento_redes(d, X_test_p, res):
    """Entrena las tres redes y devuelve la mejor (la CNN con aumento de datos)."""
    redes = {}
    for clave, config in CONFIG_REDES.items():
        print(f"Entrenando {clave}...", flush=True)
        modelo = crear_modelo(clave)
        info = entrenar(modelo, d["X_train"], d["y_train"], d["X_val"], d["y_val"], **config)
        prob = predecir_probabilidades(modelo, d["X_test"])
        prob_p = predecir_probabilidades(modelo, X_test_p)

        @torch.no_grad()
        def una(x, m=modelo):
            """Clasifica una sola imagen (para medir el tiempo por imagen)."""
            return m(torch.as_tensor(x, dtype=torch.float32)[:, None])

        res["metodos"][clave] = registro(
            NOMBRES[clave], d["y_test"], prob.argmax(1), prob_p.argmax(1), info["segundos"],
            latencia_ms(una, d["X_test"]), parametros=n_parametros(modelo),
            extra={"entrenamiento": {k: info[k] for k in ("historial", "mejor_exactitud_val", "mejor_paso", "pasos", "pasos_por_epoca")}},
        )
        redes[clave] = modelo
        imprimir(clave, res["metodos"][clave])
    return redes["cnn_aumentada"]


# --- 5. Exportación a ONNX y cadena completa --------------------------------------------


def exportar_onnx(modelo, ruta, X_prueba):
    """Exporta la red a ONNX (para la web) y devuelve la diferencia máxima con PyTorch."""
    import onnxruntime as ort

    modelo.eval()
    ejemplo = torch.zeros(1, 1, 28, 28)  # entrada de ejemplo con la forma que espera la red
    with warnings.catch_warnings():
        # el exportador clásico (dynamo=False) avisa de que quedará obsoleto; para esta red funciona igual
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        torch.onnx.export(
            modelo, (ejemplo,), str(ruta), input_names=["imagen"], output_names=["logits"],
            dynamic_axes={"imagen": {0: "lote"}, "logits": {0: "lote"}}, dynamo=False,
        )
    # Comprobación: ONNX y PyTorch deben dar las mismas puntuaciones
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    x = X_prueba[:, None].astype(np.float32)
    with torch.no_grad():
        referencia = modelo(torch.from_numpy(x)).numpy()
    return float(np.abs(referencia - sesion.run(None, {"imagen": x})[0]).max())


def medir_cadena_completa(d, n=N_SIMULADAS):
    """Exactitud del preprocesado + la red exportada con imágenes simuladas.

    A partir de dígitos del test se simulan fotos de cerca (papel con sombra, tinta,
    ruido y giro) y fotogramas de webcam (la hoja entera sobre una mesa, dentro del
    recuadro de la demo).
    """
    from .inferencia import Clasificador

    clasificador = Clasificador(MODELOS / "cnn_aumentada.onnx")
    X_s, y_s = subconjunto_estratificado(d["X_test"], d["y_test"], n)
    n = len(y_s)
    aciertos = {"fotos": 0, "webcam": 0}
    for i, (imagen, real) in enumerate(zip(X_s, y_s)):
        frame = fotograma_simulado(imagen, semilla=i)
        x0, y0, lado = recuadro_central(frame)
        casos = {
            "fotos": (foto_simulada(imagen, semilla=i), "auto"),
            "webcam": (frame[y0 : y0 + lado, x0 : x0 + lado], "claro"),
        }
        for clave, (entrada, fondo) in casos.items():
            resultado = preprocesar(entrada, fondo=fondo)
            if resultado.imagen_28 is not None:
                aciertos[clave] += int(np.argmax(clasificador.probabilidades(resultado.imagen_28)) == real)
    return {"n": n, "fotos": aciertos["fotos"] / n, "webcam": aciertos["webcam"] / n}


# --- 6. Curva de aprendizaje ------------------------------------------------------------


def curva_de_aprendizaje(d, hog, res):
    """Exactitud en test de cada método entrenado con 100 a 55.000 imágenes."""
    curva = {"tamanos": TAMANOS_CURVA, "orden": ["hog_svm", "mlp", "lenet", "cnn_aumentada"], "series": {}}
    for clave in curva["orden"]:
        curva["series"][clave] = {"nombre": NOMBRES[clave], "exactitud_test": []}

    for n in TAMANOS_CURVA:
        if n >= len(d["y_train"]):  # con todo el entrenamiento ya están los resultados de la tabla
            for clave in curva["orden"]:
                curva["series"][clave]["exactitud_test"].append(res["metodos"][clave]["test"]["exactitud"])
            continue

        Xs, ys = subconjunto_estratificado(d["X_train"], d["y_train"], n)
        Fs, _ = subconjunto_estratificado(hog["train"], d["y_train"], n)  # misma semilla: mismas imágenes
        _, _, svm_n, _ = ajustar_svm(Fs, ys, hog["val"], d["y_val"])
        curva["series"]["hog_svm"]["exactitud_test"].append(float((svm_n.predict(hog["test"]) == d["y_test"]).mean()))

        for clave in ("mlp", "lenet", "cnn_aumentada"):
            modelo = crear_modelo(clave)
            pasos_por_epoca = int(np.ceil(n / 128))
            # con pocas imágenes, al menos 1.500 pasos para que la red llegue a aprender
            entrenar(
                modelo, Xs, ys, d["X_val"], d["y_val"], epocas=CONFIG_REDES[clave]["epocas"], pasos_minimos=1500,
                evaluar_cada=max(pasos_por_epoca, 100), paciencia=5, aumentar=CONFIG_REDES[clave]["aumentar"],
                verbose=False,
            )
            acc = float((predecir_probabilidades(modelo, d["X_test"]).argmax(1) == d["y_test"]).mean())
            curva["series"][clave]["exactitud_test"].append(acc)
        print(f"  curva n={n}: " + ", ".join(f"{c}={curva['series'][c]['exactitud_test'][-1]:.3f}" for c in curva["orden"]), flush=True)
    return curva


# --- 7. Informe -------------------------------------------------------------------------


def tabla_markdown(res):
    """Tabla de resultados del README (reports/resultados.md)."""

    def pct(v):
        """0.994 -> «99,4 %»."""
        return f"{100 * v:.1f} %".replace(".", ",")

    def segundos(s):
        """Segundos de entrenamiento en texto («—» si el método no se entrena)."""
        if s <= 0:
            return "—"
        return f"{s:.0f} s" if s >= 1 else "< 1 s"

    filas = [
        "| Método | Exactitud (test) | F1 macro | Exactitud con giros y desplazamientos | Entrenamiento | Tiempo por imagen | Parámetros |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for clave in res["orden_metodos"]:
        m = res["metodos"][clave]
        parametros = f"{m['parametros']:,}".replace(",", ".") if m["parametros"] else "—"
        ms = f"{m['ms_por_imagen']:.2f} ms".replace(".", ",")
        f1 = f"{m['test']['f1_macro']:.3f}".replace(".", ",")
        filas.append(
            f"| {m['nombre']} | {pct(m['test']['exactitud'])} | {f1} | {pct(m['test_perturbado']['exactitud'])} "
            f"| {segundos(m['segundos_entrenamiento'])} | {ms} | {parametros} |"
        )
    return "\n".join(filas) + "\n"


def main():
    """Ejecuta todos los experimentos en orden y guarda los resultados."""
    inicio = time.perf_counter()
    torch.manual_seed(SEMILLA)
    d = cargar_mnist()
    X_test_p = perturbar(d["X_test"])  # el test perturbado (giros, desplazamientos y escala)
    print(f"Datos: train={len(d['y_train'])} val={len(d['y_val'])} test={len(d['y_test'])}", flush=True)

    res = {
        "fecha": date.today().isoformat(),
        "entorno": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cpu": platform.processor() or platform.machine(),
            "hilos_torch": torch.get_num_threads(),
        },
        "particion": {"train": len(d["y_train"]), "val": len(d["y_val"]), "test": len(d["y_test"])},
        "orden_metodos": list(NOMBRES),
        "metodos": {},
    }

    experimento_vecino_ssim(d, X_test_p, res)  # 1
    res["coste_ssim"] = medir_coste_ssim(d)
    experimento_plantillas(d, X_test_p, res)  # 2
    hog = experimento_hog_svm(d, X_test_p, res)  # 3
    cnn = experimento_redes(d, X_test_p, res)  # 4

    # 5. La red de la web y la demo: se guarda en PyTorch y en ONNX
    MODELOS.mkdir(exist_ok=True)
    torch.save(cnn.state_dict(), MODELOS / "cnn_aumentada.pt")
    res["onnx_diferencia_max"] = exportar_onnx(cnn, MODELOS / "cnn_aumentada.onnx", d["X_test"][:200])
    print(f"ONNX exportado (diferencia máxima con PyTorch: {res['onnx_diferencia_max']:.2e})", flush=True)
    res["cadena_completa"] = medir_cadena_completa(d)
    print(f"Cadena completa con imágenes simuladas: {res['cadena_completa']}", flush=True)

    res["curva_aprendizaje"] = curva_de_aprendizaje(d, hog, res)  # 6

    # 7. Informe y figuras
    res["segundos_totales"] = time.perf_counter() - inicio
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metricas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORTS / "resultados.md").write_text(tabla_markdown(res), encoding="utf-8")
    generar_todas(res, extras_desde_modelo())
    print(f"Listo en {res['segundos_totales'] / 60:.1f} minutos.", flush=True)


if __name__ == "__main__":
    main()
