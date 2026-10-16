"""Reproduce todos los experimentos y guarda métricas, modelo y figuras.

Uso (desde la carpeta del proyecto):

    python -m src.experimentos

Tarda algo más de media hora en un portátil con 2 núcleos y sin GPU. Escribe:

- reports/metricas.json: todas las cifras del README;
- reports/resultados.md: la tabla de resultados;
- reports/figuras/*.png: las figuras (versión clara y oscura);
- models/cnn_aumentada.pt y models/cnn_aumentada.onnx: la red que usan la web y la demo.
"""

import inspect
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
FIGURAS = REPORTS / "figuras"
MODELOS = RAIZ / "models"

N_REFERENCIAS = 1000  # referencias (100 por dígito) para la comparación directa con SSIM
REFERENCIAS_COSTE = [100, 300, 1000, 3000]
N_TEST_COSTE = 1000
ALPHAS_SVM = [1e-5, 1e-4, 1e-3]
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


def _registro(nombre, y_test, pred, y_test_p, pred_p, segundos, ms, parametros=None, extra=None):
    salida = {
        "nombre": nombre,
        "test": metricas(y_test, pred),
        "test_perturbado": metricas(y_test_p, pred_p),
        "segundos_entrenamiento": float(segundos),
        "ms_por_imagen": float(ms),
        "parametros": parametros,
    }
    if extra:
        salida.update(extra)
    return salida


def _imprimir(clave, r):
    print(
        f"[{clave}] test={r['test']['exactitud']:.4f}  perturbado={r['test_perturbado']['exactitud']:.4f}  "
        f"entreno={r['segundos_entrenamiento']:.1f}s  {r['ms_por_imagen']:.2f} ms/img",
        flush=True,
    )


def ajustar_svm(F_train, y_train, F_val, y_val):
    """Elige la regularización del SVM con el conjunto de validación.

    Devuelve (exactitud_val, alpha, modelo, segundos del entrenamiento elegido).
    """
    mejor = None
    for alpha in ALPHAS_SVM:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            # con 30 pasadas SGD a veces no llega a la tolerancia, pero la validación ya es estable
            warnings.simplefilter("ignore", ConvergenceWarning)
            modelo = crear_svm_hog(alpha).fit(F_train, y_train)
        segundos = time.perf_counter() - t0
        acc = float((modelo.predict(F_val) == y_val).mean())
        if mejor is None or acc > mejor[0]:
            mejor = (acc, alpha, modelo, segundos)
    return mejor


def medir_coste_ssim(X_tr, y_tr, X_te, y_te):
    """Exactitud y tiempo de la comparación directa con SSIM según el número de referencias.

    Se usa una muestra estratificada del test (100 imágenes por dígito), la misma en todos
    los puntos, para que la curva sea comparable con la tabla de resultados.
    """
    X_tc, y_tc = subconjunto_estratificado(X_te, y_te, N_TEST_COSTE)
    coste = {"referencias": REFERENCIAS_COSTE, "n_test": N_TEST_COSTE, "1nn": [], "media": [], "ms_por_imagen": []}
    for n_ref in REFERENCIAS_COSTE:
        Xr, yr = subconjunto_estratificado(X_tr, y_tr, n_ref)
        v = VecinoSSIM().fit(Xr, yr)
        t0 = time.perf_counter()
        Sc = v.similitudes(X_tc)
        coste["ms_por_imagen"].append((time.perf_counter() - t0) / N_TEST_COSTE * 1000)
        for modo in ("1nn", "media"):
            coste[modo].append(float((v.decidir(Sc, yr, modo) == y_tc).mean()))
        print(f"  coste SSIM refs={n_ref}: 1nn={coste['1nn'][-1]:.3f} media={coste['media'][-1]:.3f}", flush=True)
    return coste


def medir_cadena_completa(X_te, y_te, n=N_SIMULADAS):
    """Exactitud de la cadena completa (preprocesado + red exportada) con imágenes simuladas.

    A partir de dígitos del test se simulan fotos de cerca (papel con sombra, tinta, ruido y giro)
    y fotogramas de webcam (la hoja entera sobre una mesa, dentro del recuadro de la demo).
    """
    from .inferencia import Clasificador

    clasificador = Clasificador(MODELOS)
    X_s, y_s = subconjunto_estratificado(X_te, y_te, n)
    n = len(y_s)  # por si el test tiene menos imágenes que n
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
    return {"n": n, **{clave: valor / n for clave, valor in aciertos.items()}}


def exportar_onnx(modelo, ruta, X_prueba):
    """Exporta la red a ONNX (para la web) y comprueba que da lo mismo que PyTorch."""
    import onnxruntime as ort

    modelo.eval()
    ejemplo = torch.zeros(1, 1, 28, 28)
    with warnings.catch_warnings():
        # el exportador clásico (dynamo=False) avisa de que quedará obsoleto; para esta red funciona igual
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        # dynamo=False solo existe en PyTorch >= 2.5; en versiones anteriores ya es el exportador clásico
        clasico = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(
            modelo, (ejemplo,), str(ruta), input_names=["imagen"], output_names=["logits"],
            dynamic_axes={"imagen": {0: "lote"}, "logits": {0: "lote"}}, **clasico,
        )
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    x = X_prueba[:, None].astype(np.float32)
    with torch.no_grad():
        ref = modelo(torch.from_numpy(x)).numpy()
    onnx_out = sesion.run(None, {"imagen": x})[0]
    return float(np.abs(ref - onnx_out).max())


def main():
    inicio_total = time.perf_counter()
    torch.manual_seed(SEMILLA)
    d = cargar_mnist()
    X_tr, y_tr, X_val, y_val, X_te, y_te = (d[k] for k in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"))
    X_te_p = perturbar(X_te)
    print(f"Datos: train={len(y_tr)} val={len(y_val)} test={len(y_te)}", flush=True)

    res = {
        "fecha": date.today().isoformat(),
        "entorno": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cpu": platform.processor() or platform.machine(),
            "hilos_torch": torch.get_num_threads(),
        },
        "particion": {"train": len(y_tr), "val": len(y_val), "test": len(y_te)},
        "orden_metodos": list(NOMBRES),
        "metodos": {},
    }

    # --- 1. Comparación directa con muestras de referencia (SSIM) -------------------------
    X_ref, y_ref = subconjunto_estratificado(X_tr, y_tr, N_REFERENCIAS)
    vecino = VecinoSSIM().fit(X_ref, y_ref)
    t0 = time.perf_counter()
    S = vecino.similitudes(X_te)
    S_p = vecino.similitudes(X_te_p)
    segundos_ssim = time.perf_counter() - t0
    ms_vecino = latencia_ms(lambda x: vecino.predict(x), X_te, n=50)
    for clave, modo in (("media_ssim", "media"), ("vecino_ssim", "1nn")):
        res["metodos"][clave] = _registro(
            NOMBRES[clave], y_te, vecino.decidir(S, y_ref, modo), y_te, vecino.decidir(S_p, y_ref, modo),
            0.0, ms_vecino, extra={"referencias": N_REFERENCIAS, "segundos_test": segundos_ssim / 2},
        )
        _imprimir(clave, res["metodos"][clave])

    res["coste_ssim"] = medir_coste_ssim(X_tr, y_tr, X_te, y_te)

    # --- 2. Plantillas medias + SSIM -------------------------------------------------------
    t0 = time.perf_counter()
    plantillas = PlantillasSSIM().fit(X_tr, y_tr)
    segundos = time.perf_counter() - t0
    res["metodos"]["plantillas_ssim"] = _registro(
        NOMBRES["plantillas_ssim"], y_te, plantillas.predict(X_te), y_te, plantillas.predict(X_te_p),
        segundos, latencia_ms(lambda x: plantillas.predict(x), X_te),
    )
    _imprimir("plantillas_ssim", res["metodos"]["plantillas_ssim"])

    # --- 3. HOG + SVM lineal --------------------------------------------------------------
    t0 = time.perf_counter()
    F_tr = caracteristicas_hog(X_tr)
    segundos_hog_train = time.perf_counter() - t0
    F_val, F_te, F_te_p = caracteristicas_hog(X_val), caracteristicas_hog(X_te), caracteristicas_hog(X_te_p)
    acc_val, alpha, svm, segundos_svm = ajustar_svm(F_tr, y_tr, F_val, y_val)
    ms_svm = latencia_ms(lambda x: svm.predict(caracteristicas_hog(x, n_jobs=1)), X_te)
    res["metodos"]["hog_svm"] = _registro(
        NOMBRES["hog_svm"], y_te, svm.predict(F_te), y_te, svm.predict(F_te_p),
        segundos_hog_train + segundos_svm, ms_svm,
        parametros=int(svm[-1].coef_.size + svm[-1].intercept_.size),
        extra={"alpha": alpha, "exactitud_val": acc_val, "dimension_hog": int(F_tr.shape[1])},
    )
    _imprimir("hog_svm", res["metodos"]["hog_svm"])

    # --- 4. Redes neuronales ---------------------------------------------------------------
    redes = {}
    for clave, config in CONFIG_REDES.items():
        print(f"Entrenando {clave}...", flush=True)
        modelo = crear_modelo(clave)
        info = entrenar(modelo, X_tr, y_tr, X_val, y_val, **config)
        prob = predecir_probabilidades(modelo, X_te)
        prob_p = predecir_probabilidades(modelo, X_te_p)

        @torch.no_grad()
        def una(x, m=modelo):
            return m(torch.as_tensor(x, dtype=torch.float32)[:, None])

        res["metodos"][clave] = _registro(
            NOMBRES[clave], y_te, prob.argmax(1), y_te, prob_p.argmax(1), info["segundos"],
            latencia_ms(una, X_te), parametros=n_parametros(modelo),
            extra={"entrenamiento": {k: info[k] for k in ("historial", "mejor_exactitud_val", "mejor_paso", "pasos", "pasos_por_epoca")}},
        )
        redes[clave] = (modelo, prob)
        _imprimir(clave, res["metodos"][clave])

    # --- 5. Guardar la red de la demo y exportarla a ONNX ---------------------------------
    MODELOS.mkdir(exist_ok=True)
    cnn, _ = redes["cnn_aumentada"]
    torch.save(cnn.state_dict(), MODELOS / "cnn_aumentada.pt")
    res["onnx_diferencia_max"] = exportar_onnx(cnn, MODELOS / "cnn_aumentada.onnx", X_te[:200])
    print(f"ONNX exportado (diferencia máxima con PyTorch: {res['onnx_diferencia_max']:.2e})", flush=True)
    res["cadena_completa"] = medir_cadena_completa(X_te, y_te)
    print(f"Cadena completa con imágenes simuladas: {res['cadena_completa']}", flush=True)

    # --- 6. Curva de aprendizaje -----------------------------------------------------------
    curva = {"tamanos": TAMANOS_CURVA, "orden": ["hog_svm", "mlp", "lenet", "cnn_aumentada"], "series": {}}
    for clave in curva["orden"]:
        curva["series"][clave] = {"nombre": NOMBRES[clave], "exactitud_test": []}
    for n in TAMANOS_CURVA:
        if n >= len(y_tr):
            for clave in curva["orden"]:
                curva["series"][clave]["exactitud_test"].append(res["metodos"][clave]["test"]["exactitud"])
            continue
        Xs, ys = subconjunto_estratificado(X_tr, y_tr, n)
        Fs, _ = subconjunto_estratificado(F_tr, y_tr, n)  # misma semilla: mismas imágenes
        _, _, svm_n, _ = ajustar_svm(Fs, ys, F_val, y_val)
        curva["series"]["hog_svm"]["exactitud_test"].append(float((svm_n.predict(F_te) == y_te).mean()))
        for clave in ("mlp", "lenet", "cnn_aumentada"):
            modelo = crear_modelo(clave)
            ppe = int(np.ceil(n / 128))
            entrenar(
                modelo, Xs, ys, X_val, y_val, epocas=CONFIG_REDES[clave]["epocas"], pasos_minimos=1500,
                evaluar_cada=max(ppe, 100), paciencia=5, aumentar=CONFIG_REDES[clave]["aumentar"], verbose=False,
            )
            acc = float((predecir_probabilidades(modelo, X_te).argmax(1) == y_te).mean())
            curva["series"][clave]["exactitud_test"].append(acc)
        print(f"  curva n={n}: " + ", ".join(f"{c}={curva['series'][c]['exactitud_test'][-1]:.3f}" for c in curva["orden"]), flush=True)
    res["curva_aprendizaje"] = curva

    # --- 7. Informe -----------------------------------------------------------------------
    res["segundos_totales"] = time.perf_counter() - inicio_total
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metricas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (REPORTS / "resultados.md").write_text(tabla_markdown(res), encoding="utf-8")

    # figuras: las de imágenes (plantillas, fallos y preprocesado) se calculan con el modelo recién guardado
    generar_todas(res, FIGURAS, extras_desde_modelo())
    print(f"Listo en {res['segundos_totales'] / 60:.1f} minutos.", flush=True)


def _pct(v):
    return f"{100 * v:.1f} %".replace(".", ",")


def _seg(s):
    if s <= 0:
        return "—"
    return f"{s:.0f} s" if s >= 1 else "< 1 s"


def tabla_markdown(res):
    filas = [
        "| Método | Exactitud (test) | F1 macro | Exactitud con giros y desplazamientos | Entrenamiento | Tiempo por imagen | Parámetros |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for clave in res["orden_metodos"]:
        m = res["metodos"][clave]
        parametros = f"{m['parametros']:,}".replace(",", ".") if m["parametros"] else "—"
        ms = f"{m['ms_por_imagen']:.2f} ms".replace(".", ",")
        filas.append(
            f"| {m['nombre']} | {_pct(m['test']['exactitud'])} | {m['test']['f1_macro']:.3f}".replace(".", ",")
            + f" | {_pct(m['test_perturbado']['exactitud'])} | {_seg(m['segundos_entrenamiento'])} | {ms} | {parametros} |"
        )
    return "\n".join(filas) + "\n"


if __name__ == "__main__":
    main()
