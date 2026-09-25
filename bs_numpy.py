"""
Version 3 del item (b): paralelismo explicito + ajuste solo con NumPy.

Resuelve el sistema normal (X^T X) beta = X^T y directamente, sin sklearn.
Es tambien la version que el item (e) pide instrumentar para detectar
oversubscription entre los p procesos y los threads internos de BLAS.

Incluye tres variantes del ajuste para el analisis de mejoras del item (b):
  - "indices":      implementacion directa del Paso 2, materializando X[idx].
  - "pesos_densos": primer intento con pesos, X^T diag(w) X sobre la X completa.
                    Resulta MAS LENTO que "indices" (ver su docstring).
  - "pesos":        version optimizada, matematicamente equivalente, que solo
                    copia las filas sorteadas y aprovecha el producto simetrico.

Las tres sortean exactamente los mismos indices para una misma semilla, asi que
sus betas coinciden hasta el redondeo de punto flotante (~1e-14).
"""

import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info, threadpool_limits

import common


def _ajustar_indices(X, y, semilla):
    """
    Variante directa: replica literalmente el Paso 2 del enunciado.

    Sortea N indices con reemplazo, arma X_b = X[idx] e y_b = y[idx], y resuelve
    el sistema normal. Cuesta una copia de ~241 MB por resample.

    Retorna los coeficientes (k+1,).
    """
    n = X.shape[0]
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, n, size=n)
    Xb = X[idx]
    yb = y[idx]
    # Usar solve y NO np.linalg.inv: invertir explicitamente es mas caro y
    # numericamente peor condicionado que resolver el sistema.
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ yb)


def _pesos(n, semilla):
    """
    Conteos w por fila del resample: w[i] = veces que salio la fila i.

    Se sortean los mismos indices que en _ajustar_indices y se cuentan con
    bincount. Es exactamente una muestra Multinomial(n, 1/n) --- lo mismo que
    rng.multinomial(n, np.full(n, 1/n)) --- pero mucho mas barata, y con la
    ventaja de que todas las variantes usan el MISMO resample para una semilla.
    """
    rng = np.random.default_rng(semilla)
    return np.bincount(rng.integers(0, n, size=n), minlength=n)


def _ajustar_pesos_densos(X, y, semilla):
    """
    Primer intento con pesos: X^T diag(w) X calculado sobre la X completa.

    Evita el gather X[idx], pero X * w[:, None] igual crea un temporal de 241 MB
    y, peor, X.T @ Xw ya no es de la forma A.T @ A: NumPy no puede usar syrk
    (producto simetrico, la mitad de flops) y cae a un gemm general. En la
    practica es mas lento que "indices". Se deja como evidencia para el item (b).

    Retorna los coeficientes (k+1,).
    """
    w = _pesos(X.shape[0], semilla)
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw, Xw.T @ y)


def _ajustar_pesos(X, y, semilla):
    """
    Variante optimizada: reemplaza el resampleo por pesos multinomiales.

    Sortear N indices con reemplazo equivale a contar cuantas veces sale cada
    fila. Si w es ese vector de conteos, entonces:

        X_b^T X_b == X^T diag(w) X        y      X_b^T y_b == X^T (w * y)

    o sea el mismo sistema normal, pero sin materializar X_b.
    Es matematicamente identico, no una aproximacion.

    Dos detalles hacen que esta variante sea la mas rapida:
      - Solo ~63.2% de las filas sale al menos una vez (1 - 1/e); las filas con
        w=0 no aportan nada, asi que se descartan. El producto X^T X, que domina
        el costo, se hace sobre ~63k filas en vez de 100k: ~37% menos flops.
      - Se escala por sqrt(w): con Xs = sqrt(w) * X[u] queda X_b^T X_b = Xs^T Xs,
        que SI es de la forma A.T @ A y NumPy lo resuelve con syrk.

    Retorna los coeficientes (k+1,).
    """
    w = _pesos(X.shape[0], semilla)
    u = np.flatnonzero(w)                 # filas que salieron al menos una vez
    s = np.sqrt(w[u])
    Xs = X[u]                             # copia de ~63% de X (no 100%)
    Xs *= s[:, None]                      # in-place: sin segundo temporal
    return np.linalg.solve(Xs.T @ Xs, Xs.T @ (s * y[u]))


# Registro de variantes, para que bench.py pueda barrerlas y comparar.
VARIANTES = {
    "indices": _ajustar_indices,
    "pesos_densos": _ajustar_pesos_densos,
    "pesos": _ajustar_pesos,
}


def _tarea(fn, X, y, semilla, threads):
    """
    Envoltorio que ejecuta un resample dentro del worker con t threads de BLAS.

    El limite se aplica AQUI, dentro del proceso hijo: con 'spawn' el hijo
    arranca con su propio OpenBLAS/MKL y no hereda un threadpool_limits hecho
    en el padre. Con threads=None no se toca nada (queda el default del worker).
    """
    if threads is None:
        return fn(X, y, semilla)
    with threadpool_limits(limits=threads, user_api="blas"):
        return fn(X, y, semilla)


def run(X, y, p, b=common.B, seed=common.SEED, variante="pesos", threads=None,
        backend="loky"):
    """
    Reparte los b resamples entre p procesos y retorna los coeficientes (b, k+1).

    'threads' controla cuantos hilos usa BLAS DENTRO de cada proceso. Es el
    parametro t del item (i): se busca la mejor combinacion (p, t) sujeta a
    p * t <= cores logicos. Con threads=None cada proceso intenta usar todos los
    cores, lo que produce la oversubscription que el item (e) pide diagnosticar.

    Ojo: el backend 'loky' (default de joblib) ya limita por su cuenta los
    threads de BLAS de cada worker a cpu_count // p. La oversubscription "cruda"
    solo aparece con backend='multiprocessing' y threads=None.
    """
    fn = VARIANTES[variante]
    semillas = common.semillas_resamples(seed, b)
    # Parallel devuelve los resultados en el orden de las tareas (no en el orden
    # en que terminan), asi que la fila i siempre corresponde a semillas[i].
    betas = Parallel(n_jobs=p, backend=backend, batch_size=1)(
        delayed(_tarea)(fn, X, y, s, threads) for s in semillas
    )
    return np.vstack(betas)


# ---------------------------------------------------------------------------
# Item (e): diagnostico de oversubscription
# ---------------------------------------------------------------------------

def _reportar_threads(threads=None):
    """
    Tarea auxiliar que se ejecuta dentro de un worker y reporta su estado de threads.

    Retorna el PID mas la salida de threadpool_info(), que lista cada libreria
    nativa cargada (OpenBLAS / MKL) y cuantos threads tiene configurados.
    Si se pasa 'threads', el reporte se toma con ese limite aplicado.
    """
    # La pausa evita que un mismo worker (ya libre) se lleve varias de estas
    # tareas instantaneas: asi cada uno de los p procesos reporta la suya.
    time.sleep(0.5)
    if threads is None:
        return {"pid": os.getpid(), "pools": threadpool_info()}
    with threadpool_limits(limits=threads, user_api="blas"):
        return {"pid": os.getpid(), "pools": threadpool_info()}


def inspeccionar_threads(p, threads=None, backends=("loky", "multiprocessing")):
    """
    Lanza p workers y muestra cuantos threads usa NumPy internamente en cada uno.

    Evidencia buscada para el item (e): si cada uno de los p procesos reporta
    num_threads igual al total de cores logicos, hay oversubscription
    (p * t hilos compitiendo por 4 cores). La correccion es fijar t con
    threadpool_limits de modo que p * t <= cores logicos.

    Se inspeccionan ambos backends porque se comportan distinto: 'loky' limita
    solo los threads de sus workers, 'multiprocessing' no.
    """
    cores = os.cpu_count()
    print(f"Cores logicos: {cores}   p={p}   limite t={threads}")
    print(f"Proceso padre (pid={os.getpid()}):")
    for pool in threadpool_info():
        print(f"    {pool['user_api']:7s} {pool['internal_api']:9s} "
              f"num_threads={pool['num_threads']}")

    for backend in backends:
        reportes = Parallel(n_jobs=p, backend=backend, batch_size=1)(
            delayed(_reportar_threads)(threads) for _ in range(p)
        )
        print(f"\nBackend '{backend}':")
        por_pid = {r["pid"]: r["pools"] for r in reportes}
        total = 0
        for pid, pools in sorted(por_pid.items()):
            for pool in pools:
                print(f"  worker pid={pid:6d}  {pool['user_api']:7s} "
                      f"{pool['internal_api']:9s} num_threads={pool['num_threads']}")
                if pool["user_api"] == "blas":
                    total += pool["num_threads"]
        estado = "OVERSUBSCRIPTION" if total > cores else "ok"
        print(f"  -> {len(por_pid)} workers con {total} threads BLAS en total "
              f"para {cores} cores logicos: {estado}")


def main():
    """
    CLI de esta version.

    Flags previstos:
      -p           numero de procesos
      --variante   indices | pesos_densos | pesos
      -t           threads internos por proceso (item i)
      --backend    loky | multiprocessing
      --threads-info  solo ejecuta inspeccionar_threads() para el item (e)
      --repetir    repite la corrida n veces, para observarla en el monitor
    """
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-p", type=int, default=1, help="numero de procesos")
    parser.add_argument("--variante", choices=list(VARIANTES), default="pesos")
    parser.add_argument("-t", type=int, default=None,
                        help="threads de BLAS por proceso (default: sin limite)")
    parser.add_argument("--backend", choices=["loky", "multiprocessing"], default="loky")
    parser.add_argument("--threads-info", action="store_true",
                        help="solo inspecciona los threads de cada worker (item e)")
    parser.add_argument("--repetir", type=int, default=1,
                        help="repite la corrida n veces (util para mirar el monitor)")
    args = parser.parse_args()

    if args.threads_info:
        inspeccionar_threads(args.p, args.t)
        return

    X, y, beta_true = common.cargar_datos()
    for _ in range(args.repetir):
        t0 = time.perf_counter()
        betas = run(X, y, args.p, variante=args.variante, threads=args.t,
                    backend=args.backend)
        tiempo = time.perf_counter() - t0
        lo, hi = common.intervalo_confianza(betas)
        cobertura = np.mean((lo <= beta_true) & (beta_true <= hi))
        print(f"bs_numpy  p={args.p}  t={args.t}  variante={args.variante}  "
              f"backend={args.backend}  tiempo={tiempo:.2f} s  "
              f"cobertura IC95={cobertura:.3f}")
    for j in range(3):
        print(f"  beta_{j}: IC=[{lo[j]:+.4f}, {hi[j]:+.4f}]  verdadero={beta_true[j]:+.4f}")


if __name__ == "__main__":
    # Guard obligatorio en Windows (backend 'spawn' re-importa el modulo).
    main()
