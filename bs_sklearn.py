"""
Version 2 del item (b): paralelismo explicito + ajuste con sklearn.

El bucle sobre los B resamples y su reparto entre p procesos lo escribimos
nosotros con joblib.Parallel; el ajuste de minimos cuadrados queda en manos de
LinearRegression. Frente a bs_auto.py ganamos control sobre como se generan los
indices y como se despachan las tareas.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

import common


def _ajustar_resample(X, y, semilla):
    """
    Tarea unitaria que ejecuta un worker: un resample completo (Paso 2, i-iii).

    Recibe su propia semilla (SeedSequence) en vez de compartir un RNG global,
    de modo que el resultado no dependa de p ni del orden de despacho.

    Retorna el vector de coeficientes (k+1,) de ese resample.
    """
    # TODO: rng = np.random.default_rng(semilla)
    # TODO: idx = rng.integers(0, X.shape[0], size=X.shape[0])   # N indices con reemplazo
    # TODO: LinearRegression(fit_intercept=False).fit(X[idx], y[idx]).coef_
    #
    # NOTA para el item (b) "comente sus mejoras": X[idx] materializa una copia
    # de ~241 MB por resample. Es el costo dominante de esta version y explica
    # por que escala peor que bs_numpy.py con pesos.
    raise NotImplementedError


def run(X, y, p, b=common.B, seed=common.SEED):
    """
    Reparte los b resamples entre p procesos y retorna los coeficientes (b, k+1).

    Parametros de Parallel que conviene justificar en el informe:
      - backend: 'loky' (default) vs 'multiprocessing'; el item (d) pide describir
        este ultimo, asi que vale la pena medir ambos.
      - batch_size: con B=48 tareas gruesas conviene 1, para no desbalancear.
      - max_nbytes: umbral sobre el cual joblib hace memory-mapping de los
        arreglos en vez de serializarlos hacia cada worker (clave en Windows,
        donde 'spawn' no permite copy-on-write).
    """
    # TODO: semillas = common.semillas_resamples(seed, b)
    # TODO: Parallel(n_jobs=p)(delayed(_ajustar_resample)(X, y, s) for s in semillas)
    # TODO: np.vstack del resultado
    raise NotImplementedError


def main():
    """Corrida individual de esta version."""
    # TODO: cargar datos, correr run() para un p dado, imprimir el IC y el tiempo.
    pass


if __name__ == "__main__":
    # Guard obligatorio en Windows (backend 'spawn' re-importa el modulo).
    main()
