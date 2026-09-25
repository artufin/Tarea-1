"""
Base compartida por todas las versiones del bootstrap.

Cubre el item (a) del enunciado (generacion de datos sinteticos) y el Paso 3
del algoritmo (construccion del intervalo de confianza), ademas de utilidades
de cronometraje y registro de resultados que usan bench.py, check.py y plots.py.
"""

import csv
import os
import platform
import time

import numpy as np

# --- Parametros del experimento (fijos por enunciado) ---
N = 100_000          # observaciones
K = 300              # variables de entrada (X queda de N x (K+1) con la columna de unos)
B = 48               # resamples bootstrap
SEED = 12345         # semilla maestra: fija la generacion de datos Y los resamples

# --- Rutas ---
DIR_DATOS = "datos"
DIR_RESULTADOS = "resultados"
CSV_TIEMPOS = os.path.join(DIR_RESULTADOS, "tiempos.csv")

# Columnas del CSV de tiempos. Incluye 'maquina' porque el item (j) exige
# comparar los resultados de dos computadores distintos.
COLUMNAS_CSV = [
    "maquina", "version", "variante", "p", "t",
    "N", "k", "B", "repeticion", "tiempo_s",
]


def nombre_maquina():
    """Identificador corto del computador actual, para etiquetar resultados (item j)."""
    return platform.node()


def info_maquina():
    """
    Ficha tecnica del equipo para incluir en el informe: CPU, cores logicos, SO.

    El item (f) pide declarar explicitamente el numero de cores logicos, y el
    item (j) necesita estos datos para justificar las diferencias entre maquinas.
    """
    from threadpoolctl import threadpool_info

    blas = [
        f"{pool.get('internal_api')} {pool.get('version')} "
        f"({pool.get('num_threads')} threads, {pool.get('threading_layer', '-')})"
        for pool in threadpool_info()
        if pool.get("user_api") == "blas"
    ]
    return {
        "maquina": nombre_maquina(),
        "procesador": platform.processor() or platform.machine(),
        "cores_logicos": os.cpu_count(),
        "so": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": "; ".join(blas) or "desconocido",
    }


# ---------------------------------------------------------------------------
# Item (a): generacion de datos sinteticos
# ---------------------------------------------------------------------------

def generar_datos(n=N, k=K, seed=SEED):
    """
    Genera (X, y, beta_true) segun el item (a), con semilla fija.

    (i)   beta_true: k+1 coeficientes i.i.d. N(0,1)
    (ii)  X: matriz n x k con entradas i.i.d. N(0,1), mas una columna de unos
          antepuesta -> queda de n x (k+1)
    (iii) y = X @ beta_true + ruido, con ruido i.i.d. N(0,1) de largo n

    Retorna X (n, k+1), y (n,), beta_true (k+1,).
    """
    rng = np.random.default_rng(seed)

    # (i) coeficientes verdaderos
    beta_true = rng.standard_normal(k + 1)

    # (ii) X se reserva una sola vez y se llena por bloques de filas: asi no se
    # materializa un temporal n x k aparte (ni un hstack) que duplique la memoria.
    X = np.empty((n, k + 1))
    X[:, 0] = 1.0
    bloque = 10_000
    for i in range(0, n, bloque):
        j = min(i + bloque, n)
        X[i:j, 1:] = rng.standard_normal((j - i, k))

    # (iii) salida con ruido
    y = X @ beta_true + rng.standard_normal(n)
    return X, y, beta_true


def cargar_datos(regenerar=False):
    """
    Devuelve (X, y, beta_true) leyendo de cache en disco, generandolos si no existen.

    Cachear es importante por dos razones:
      - las tres versiones deben correr sobre datos identicos para que el item (c)
        pueda comparar intervalos de confianza entre si;
      - regenerar 241 MB en cada corrida contaminaria los tiempos del item (f).
    """
    rutas = {nombre: os.path.join(DIR_DATOS, f"{nombre}.npy")
             for nombre in ("X", "y", "beta_true")}

    if regenerar or not all(os.path.exists(r) for r in rutas.values()):
        os.makedirs(DIR_DATOS, exist_ok=True)
        X, y, beta_true = generar_datos()
        np.save(rutas["X"], X)
        np.save(rutas["y"], y)
        np.save(rutas["beta_true"], beta_true)
        del X, y, beta_true

    # X se abre como memmap de solo lectura: joblib reconoce np.memmap y a los
    # workers les envia solo la ruta del archivo (no los 241 MB serializados),
    # y todos los procesos terminan compartiendo las mismas paginas del cache
    # de disco del SO. y y beta_true son chicos y se cargan en RAM.
    X = np.load(rutas["X"], mmap_mode="r")
    y = np.load(rutas["y"])
    beta_true = np.load(rutas["beta_true"])
    return X, y, beta_true


# ---------------------------------------------------------------------------
# Paso 3 del algoritmo: intervalo de confianza
# ---------------------------------------------------------------------------

def intervalo_confianza(betas, nivel=0.95):
    """
    Intervalo de confianza bootstrap por percentiles (Paso 3).

    Recibe betas de forma (B, k+1) --- un vector de coeficientes por resample ---
    ordena cada coeficiente j y descarta 2.5% por lado.

    Retorna (lo, hi), ambos de forma (k+1,).
    """
    cola = (1.0 - nivel) / 2.0 * 100.0
    lo, hi = np.percentile(np.asarray(betas), [cola, 100.0 - cola], axis=0)
    return lo, hi


def semillas_resamples(seed=SEED, b=B):
    """
    Genera b semillas independientes, una por resample.

    Clave para el item (c): si cada tarea recibe su propia semilla derivada de
    forma deterministica, el resultado no depende de p ni del orden en que el
    scheduler despache las tareas. Compartir un unico RNG global entre procesos
    romperia la reproducibilidad.
    """
    return np.random.SeedSequence(seed).spawn(b)


# ---------------------------------------------------------------------------
# Cronometraje y registro
# ---------------------------------------------------------------------------

def cronometrar(fn, repeticiones=3, descartar_warmup=True):
    """
    Ejecuta fn() varias veces y retorna la lista de tiempos en segundos.

    Se usa time.perf_counter (reloj monotonico de alta resolucion). Se descarta
    la primera corrida porque incluye costos unicos: import de sklearn en los
    workers, creacion del pool de joblib y primer toque de las paginas de memoria.
    Reportar la MEDIANA y no el promedio, porque este equipo es un notebook y el
    throttling termico produce outliers hacia arriba.
    """
    if descartar_warmup:
        fn()
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        tiempos.append(time.perf_counter() - t0)
    return tiempos


def registrar_tiempo(version, variante, p, t, repeticion, tiempo_s, csv_path=CSV_TIEMPOS):
    """
    Agrega una fila al CSV de resultados, creando el header si el archivo no existe.

    Una fila por (version, variante, p, t, repeticion) para no perder la dispersion
    de las mediciones al agregar despues en plots.py.
    """
    carpeta = os.path.dirname(csv_path)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    nuevo = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        if nuevo:
            writer.writeheader()
        writer.writerow({
            "maquina": nombre_maquina(),
            "version": version,
            "variante": variante,
            "p": p,
            "t": t,
            "N": N,
            "k": K,
            "B": B,
            "repeticion": repeticion,
            "tiempo_s": f"{tiempo_s:.6f}",
        })


def leer_tiempos(csv_path=CSV_TIEMPOS):
    """Lee el CSV de tiempos y lo retorna como lista de dicts (lo consume plots.py)."""
    filas = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            for col in ("p", "t", "N", "k", "B", "repeticion"):
                fila[col] = int(fila[col])
            fila["tiempo_s"] = float(fila["tiempo_s"])
            filas.append(fila)
    return filas


if __name__ == "__main__":
    # Ejecutar este archivo directamente genera y cachea los datos del item (a).
    t0 = time.perf_counter()
    X, y, beta_true = cargar_datos(regenerar=True)
    print(f"Datos generados en {time.perf_counter() - t0:.1f} s (semilla {SEED})")
    print(f"  X: {X.shape} {X.dtype}  ({X.nbytes / 1e6:.0f} MB)")
    print(f"  y: {y.shape}   beta_true: {beta_true.shape}")
    print("Ficha de la maquina:")
    for clave, valor in info_maquina().items():
        print(f"  {clave:14s} {valor}")
