"""Métodos clásicos (sin redes neuronales).

1. Plantillas medias + SSIM: una plantilla por dígito (la media de sus imágenes de
   entrenamiento); cada imagen nueva se asigna al dígito cuya plantilla se le parece más.
2. Comparación directa con SSIM: se compara la imagen con muchas imágenes de referencia
   y se elige el dígito de la más parecida (o el de mayor parecido medio).
3. HOG + SVM lineal: se describe cada imagen por la dirección de sus bordes (HOG) y un
   clasificador lineal (SVM) aprende a separar los dígitos con esa descripción.
"""

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datos import SEMILLA
from .ssim import ssim_por_pares

N_CLASES = 10  # dígitos del 0 al 9


class PlantillasSSIM:
    """Una plantilla por dígito: la imagen media de todas sus imágenes de entrenamiento."""

    def fit(self, X, y):
        """Calcula las 10 plantillas: plantillas_[c] es la media de las imágenes del dígito c."""
        self.plantillas_ = np.stack([X[y == c].mean(axis=0) for c in range(N_CLASES)])
        return self

    def predict(self, X, bloque=2000):
        """SSIM de cada imagen con las 10 plantillas; gana la de mayor SSIM."""
        etiquetas = []
        for i in range(0, len(X), bloque):
            similitud = ssim_por_pares(X[i : i + bloque], self.plantillas_)  # forma (bloque, 10)
            etiquetas.append(similitud.argmax(axis=1))
        return np.concatenate(etiquetas)


class VecinoSSIM:
    """Comparación directa con imágenes de referencia usando SSIM.

    Hay dos formas de decidir a partir de las similitudes:
    - "1nn": el dígito de la referencia más parecida (vecino más cercano);
    - "media": el dígito cuyas referencias tienen mayor SSIM medio.
    """

    def fit(self, X, y):
        """No hay entrenamiento: solo se guardan las imágenes de referencia."""
        self.X_ref_, self.y_ref_ = X, y
        return self

    def similitudes(self, X, bloque=1000):
        """Matriz de SSIM entre cada imagen de X y cada referencia (es lo que más tarda)."""
        partes = [ssim_por_pares(X[i : i + bloque], self.X_ref_) for i in range(0, len(X), bloque)]
        return np.concatenate(partes)

    @staticmethod
    def decidir(similitud, y_ref, modo):
        """Elige el dígito de cada imagen a partir de su fila de similitudes."""
        if modo == "1nn":
            return y_ref[similitud.argmax(axis=1)]
        # modo "media": SSIM medio con las referencias de cada dígito y el mayor de los 10
        medias = np.stack([similitud[:, y_ref == c].mean(axis=1) for c in range(N_CLASES)], axis=1)
        return medias.argmax(axis=1)

    def predict(self, X, modo="1nn"):
        """Dígito de cada imagen según el modo elegido ("1nn" o "media")."""
        return self.decidir(self.similitudes(X), self.y_ref_, modo)


def caracteristicas_hog(X, celda=4, n_jobs=2):
    """Describe cada imagen con HOG (histogramas de gradientes orientados).

    La imagen se divide en celdas de 4x4 píxeles y en cada celda se cuenta en qué
    direcciones van los bordes (9 direcciones). Con imágenes de 28x28 salen 1.296
    valores por imagen. Se calcula en paralelo con 2 procesos para ir más rápido.
    """

    def una(imagen):
        """HOG de una sola imagen."""
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
    """SVM lineal sobre las características HOG.

    Primero se estandariza cada característica (media 0 y desviación 1) y después se
    entrena un SVM lineal con descenso de gradiente estocástico (SGDClassifier con
    pérdida hinge). Con 55.000 imágenes tarda segundos. «alpha» es la regularización:
    se elige con el conjunto de validación.
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", alpha=alpha, max_iter=30, tol=1e-4, random_state=SEMILLA, n_jobs=2),
    )
