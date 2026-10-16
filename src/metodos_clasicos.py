"""Métodos clásicos: comparación de plantillas con SSIM y HOG + SVM lineal."""

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datos import SEMILLA
from .ssim import ssim_por_pares

N_CLASES = 10


class PlantillasSSIM:
    """Una plantilla por dígito (la imagen media de sus muestras de entrenamiento).

    Cada imagen se asigna al dígito cuya plantilla tiene mayor SSIM con ella.
    """

    def fit(self, X, y):
        self.plantillas_ = np.stack([X[y == c].mean(axis=0) for c in range(N_CLASES)])
        return self

    def predict(self, X, bloque=2000):
        etiquetas = []
        for i in range(0, len(X), bloque):
            similitud = ssim_por_pares(X[i : i + bloque], self.plantillas_)
            etiquetas.append(similitud.argmax(axis=1))
        return np.concatenate(etiquetas)


class VecinoSSIM:
    """Comparación directa con muestras de referencia usando SSIM.

    - modo "1nn": el dígito de la muestra de referencia más parecida.
    - modo "media": el dígito con mayor SSIM medio entre sus referencias.
    """

    def fit(self, X, y):
        self.X_ref_, self.y_ref_ = X, y
        return self

    def similitudes(self, X, bloque=1000):
        partes = [ssim_por_pares(X[i : i + bloque], self.X_ref_) for i in range(0, len(X), bloque)]
        return np.concatenate(partes)

    @staticmethod
    def decidir(similitud, y_ref, modo):
        if modo == "1nn":
            return y_ref[similitud.argmax(axis=1)]
        medias = np.stack([similitud[:, y_ref == c].mean(axis=1) for c in range(N_CLASES)], axis=1)
        return medias.argmax(axis=1)

    def predict(self, X, modo="1nn"):
        return self.decidir(self.similitudes(X), self.y_ref_, modo)


def caracteristicas_hog(X, celda=4, n_jobs=2):
    """Histogramas de gradientes orientados (HOG) de cada imagen: 1.296 valores con celdas de 4x4."""

    def una(imagen):
        return hog(
            imagen,
            orientations=9,
            pixels_per_cell=(celda, celda),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

    filas = Parallel(n_jobs=n_jobs, batch_size=512)(delayed(una)(imagen) for imagen in X)
    return np.asarray(filas, dtype=np.float32)


def crear_svm_hog(alpha=1e-5):
    """SVM lineal (uno contra el resto) sobre HOG estandarizado, entrenado con descenso de gradiente
    estocástico (pérdida hinge). Con 55.000 imágenes tarda segundos, frente a minutos de LinearSVC.
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", alpha=alpha, max_iter=30, tol=1e-4, random_state=SEMILLA, n_jobs=2),
    )
