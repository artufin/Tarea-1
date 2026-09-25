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
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)   # N indices con reemplazo
    # NOTA para el item (b) "comente sus mejoras": X[idx] materializa una copia
    # de ~241 MB por resample. Es el costo dominante de esta version y explica
    # por que escala peor que bs_numpy.py con pesos. copy_X=False evita que
    # LinearRegression haga una SEGUNDA copia de esa matriz (ya es nuestra).
    modelo = LinearRegression(fit_intercept=False, copy_X=False)
    return modelo.fit(X[idx], y[idx]).coef_


def _ajustar_resample_pesos(X, y, semilla):
    """
    Mejora sobre _ajustar_resample: mismos indices, pero pasados como pesos.

    Con la misma semilla sortea exactamente los mismos indices y los convierte
    en conteos por fila (w = bincount(idx)), que LinearRegression recibe como
    sample_weight. El problema de minimos cuadrados ponderado es identico al del
    resample, y ademas las filas con w=0 (~36.8% de ellas) se descartan antes de
    ajustar, asi que la matriz que llega a lstsq es ~37% mas chica que X[idx].
    Es lo mismo que hace BaggingRegressor internamente (sin descartar filas).
    """
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    idx = rng.integers(0, n, size=n)
    w = np.bincount(idx, minlength=n)
    usadas = np.flatnonzero(w)
    modelo = LinearRegression(fit_intercept=False, copy_X=False)
    return modelo.fit(X[usadas], y[usadas], sample_weight=w[usadas]).coef_


# Registro de variantes, para que bench.py pueda compararlas (item b).
VARIANTES = {
    "indices": _ajustar_resample,
    "pesos": _ajustar_resample_pesos,
}


def run(X, y, p, b=common.B, seed=common.SEED, variante="pesos", backend="loky"):
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
    fn = VARIANTES[variante]
    semillas = common.semillas_resamples(seed, b)
    # Parallel devuelve los resultados en el orden de las tareas (no en el orden
    # en que terminan), asi que la fila i siempre corresponde a semillas[i].
    betas = Parallel(n_jobs=p, backend=backend, batch_size=1)(
        delayed(fn)(X, y, s) for s in semillas
    )
    return np.vstack(betas)


def main():
    """Corrida individual de esta version."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-p", type=int, default=1, help="numero de procesos")
    parser.add_argument("--variante", choices=list(VARIANTES), default="pesos")
    parser.add_argument("--backend", choices=["loky", "multiprocessing"], default="loky")
    args = parser.parse_args()

    X, y, beta_true = common.cargar_datos()
    t0 = time.perf_counter()
    betas = run(X, y, args.p, variante=args.variante, backend=args.backend)
    tiempo = time.perf_counter() - t0

    lo, hi = common.intervalo_confianza(betas)
    cobertura = np.mean((lo <= beta_true) & (beta_true <= hi))
    print(f"bs_sklearn  p={args.p}  variante={args.variante}  backend={args.backend}  "
          f"tiempo={tiempo:.2f} s  cobertura IC95={cobertura:.3f}")
    for j in range(3):
        print(f"  beta_{j}: IC=[{lo[j]:+.4f}, {hi[j]:+.4f}]  verdadero={beta_true[j]:+.4f}")


if __name__ == "__main__":
    # Guard obligatorio en Windows (backend 'spawn' re-importa el modulo).
    main()
