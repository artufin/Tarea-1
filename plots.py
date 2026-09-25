"""
Generacion de todas las figuras del informe a partir del CSV de tiempos.

Cubre los items (f), (g), (h), (i) y (j). Se separa de bench.py a proposito:
medir toma varios minutos, graficar toma segundos, y el enunciado evalua
explicitamente la claridad de graficos y tablas (titulos, labels, leyendas y
tamano de fuente adecuados).

Uso:
    python plots.py                 # todas las figuras de esta maquina
    python plots.py --maquina X     # figuras de otra maquina presente en el CSV
    python plots.py --comparar      # item (j), requiere el CSV de ambos equipos

Las figuras quedan en figuras/<nombre>_<maquina>.png y las tablas de T, S, E y
To se imprimen por consola (listas para pegar en el informe).
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")   # solo se guardan archivos; no hace falta ventana
import matplotlib.pyplot as plt
import numpy as np

import common

DIR_FIGURAS = "figuras"

# Estilo comun a todas las figuras, para que el informe se vea homogeneo.
# El enunciado evalua legibilidad, asi que conviene subir el tamano de fuente
# respecto del default de matplotlib.
ESTILO = {
    "figure.figsize": (7, 4.5),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": True,
}

# Variante con que bench.py --barrido corre cada version (bench.VARIANTE_POR_VERSION).
# Se repite aqui para no importar sklearn solo para graficar.
VARIANTE_POR_VERSION = {"auto": "bagging", "sklearn": "pesos", "numpy": "pesos"}

ETIQUETAS = {
    "auto": "auto (BaggingRegressor)",
    "sklearn": "sklearn (joblib + LinearRegression)",
    "numpy": "numpy (joblib + NumPy)",
}
COLORES = {"auto": "tab:blue", "sklearn": "tab:orange", "numpy": "tab:green"}
MARCADORES = {"auto": "o", "sklearn": "s", "numpy": "^"}


def agregar_tiempos(filas):
    """
    Agrupa las filas crudas del CSV y retorna la mediana por combinacion.

    Colapsa las repeticiones en un solo valor por (maquina, version, variante, p, t).
    Se usa mediana y no promedio por el throttling termico del notebook.

    Retorna dict {clave: {"mediana", "min", "max", "n"}}.
    """
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["maquina"], f["version"], f["variante"], f["p"], f["t"])].append(f["tiempo_s"])
    return {
        clave: {
            "mediana": float(np.median(ts)),
            "min": float(np.min(ts)),
            "max": float(np.max(ts)),
            "n": len(ts),
        }
        for clave, ts in grupos.items()
    }


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _serie(datos, maquina, version, variante=None, t=0):
    """T(p) de una version: arrays (p, mediana, min, max) ordenados por p."""
    variante = variante or VARIANTE_POR_VERSION[version]
    puntos = sorted(
        (clave[3], v) for clave, v in datos.items()
        if clave[0] == maquina and clave[1] == version
        and clave[2] == variante and clave[4] == t
    )
    if not puntos:
        return None
    p = np.array([q for q, _ in puntos])
    med = np.array([v["mediana"] for _, v in puntos])
    lo = np.array([v["min"] for _, v in puntos])
    hi = np.array([v["max"] for _, v in puntos])
    return p, med, lo, hi


def _series_barrido(datos, maquina):
    """Series T(p) de las versiones presentes en el barrido de una maquina."""
    series = {}
    for version in VARIANTE_POR_VERSION:
        s = _serie(datos, maquina, version)
        if s is not None and 1 in s[0]:
            series[version] = s
    return series


def _metricas(p, T):
    """S(p), E(p) y To(p) usando el T(1) de la propia serie."""
    T1 = T[p == 1][0]
    S = T1 / T
    E = S / p
    To = p * T - T1
    return S, E, To


def _titulo(texto, maquina, p_max=None):
    extra = f", {p_max} cores logicos" if p_max else ""
    n = f"{common.N:_}".replace("_", ".")
    return f"{texto}\nN={n}, k={common.K}, B={common.B} - {maquina}{extra}"


def _guardar(fig, nombre, maquina):
    ruta = os.path.join(DIR_FIGURAS, f"{nombre}_{maquina}.png")
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


def _p_max(series):
    return int(max(s[0].max() for s in series.values()))


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def plot_tiempos(datos, maquina):
    """
    Item (f): T(p) para las tres versiones en un mismo grafico.

    Eje x = numero de procesos p, eje y = tiempo en segundos. Una curva por
    version. Conviene anotar en el titulo N, k, B y el numero de cores logicos.
    Las barras de error muestran el rango (min-max) de las repeticiones.
    """
    series = _series_barrido(datos, maquina)
    if not series:
        print("  (sin datos de --barrido: se omite plot_tiempos)")
        return
    p_max = _p_max(series)

    fig, ax = plt.subplots()
    for version, (p, med, lo, hi) in series.items():
        ax.errorbar(p, med, yerr=[med - lo, hi - med], marker=MARCADORES[version],
                    color=COLORES[version], capsize=3, label=ETIQUETAS[version])
    ax.set_xlabel("Procesos p")
    ax.set_ylabel("Tiempo de ejecucion [s]")
    ax.set_xticks(range(1, p_max + 1))
    ax.set_ylim(bottom=0)
    ax.set_title(_titulo("Tiempo de ejecucion T(p)", maquina, p_max))
    ax.legend()
    _guardar(fig, "f_tiempos", maquina)

    # Misma figura en escala log: las versiones difieren en un orden de magnitud
    # y en escala lineal la curva de numpy queda aplastada contra el eje.
    fig, ax = plt.subplots()
    for version, (p, med, lo, hi) in series.items():
        ax.errorbar(p, med, yerr=[med - lo, hi - med], marker=MARCADORES[version],
                    color=COLORES[version], capsize=3, label=ETIQUETAS[version])
    ax.set_yscale("log")
    ax.set_xlabel("Procesos p")
    ax.set_ylabel("Tiempo de ejecucion [s] (escala log)")
    ax.set_xticks(range(1, p_max + 1))
    ax.set_title(_titulo("Tiempo de ejecucion T(p)", maquina, p_max))
    ax.legend()
    _guardar(fig, "f_tiempos_log", maquina)


def plot_speedup(datos, maquina):
    """
    Item (g): S(p) = T(1)/T(p) para las tres versiones, contra la curva ideal.

    Sobre la eleccion de T(1), que el enunciado pide justificar: usar el T(1) de
    CADA version mide su propia escalabilidad, pero premia a una version lenta
    en serie. Usar el T(1) de la version mas rapida da el speedup absoluto y
    permite comparar versiones entre si. Lo mas defendible es graficar el
    relativo y reportar en el texto el mejor T(1) global.

    Se generan ambas figuras: g_speedup (relativo) y g_speedup_absoluto.
    """
    series = _series_barrido(datos, maquina)
    if not series:
        return
    p_max = _p_max(series)
    ideal = np.arange(1, p_max + 1)

    fig, ax = plt.subplots()
    ax.plot(ideal, ideal, "--", color="gray", label="Ideal S(p) = p")
    for version, (p, med, _, _) in series.items():
        S, _, _ = _metricas(p, med)
        ax.plot(p, S, marker=MARCADORES[version], color=COLORES[version],
                label=ETIQUETAS[version])
    ax.set_xlabel("Procesos p")
    ax.set_ylabel("Speedup S(p) = T(1) / T(p)")
    ax.set_xticks(ideal)
    ax.set_ylim(0, p_max + 0.5)
    ax.set_title(_titulo("Speedup relativo (T(1) de cada version)", maquina, p_max))
    ax.legend()
    _guardar(fig, "g_speedup", maquina)

    # Speedup absoluto: todas contra el T(1) de la version serial mas rapida.
    mejor = min(series, key=lambda v: series[v][1][series[v][0] == 1][0])
    T1_ref = series[mejor][1][series[mejor][0] == 1][0]
    fig, ax = plt.subplots()
    ax.plot(ideal, ideal, "--", color="gray", label="Ideal S(p) = p")
    for version, (p, med, _, _) in series.items():
        ax.plot(p, T1_ref / med, marker=MARCADORES[version], color=COLORES[version],
                label=ETIQUETAS[version])
    ax.set_xlabel("Procesos p")
    ax.set_ylabel(f"Speedup absoluto = T1[{mejor}] / T(p)")
    ax.set_xticks(ideal)
    ax.set_yscale("log")
    ax.set_title(_titulo(f"Speedup absoluto (T(1) de '{mejor}' = {T1_ref:.2f} s)",
                         maquina, p_max))
    ax.legend()
    _guardar(fig, "g_speedup_absoluto", maquina)


def plot_eficiencia(datos, maquina):
    """
    Item (g): E(p) = S(p)/p, con la referencia ideal E(p)=1.

    Fijar ylim en (0, 1.05) para que la caida de eficiencia sea legible y no
    quede aplastada por algun outlier.
    """
    series = _series_barrido(datos, maquina)
    if not series:
        return
    p_max = _p_max(series)

    fig, ax = plt.subplots()
    ax.axhline(1.0, ls="--", color="gray", label="Ideal E(p) = 1")
    e_max = 1.0
    for version, (p, med, _, _) in series.items():
        _, E, _ = _metricas(p, med)
        e_max = max(e_max, E.max())
        ax.plot(p, E, marker=MARCADORES[version], color=COLORES[version],
                label=ETIQUETAS[version])
    ax.set_xlabel("Procesos p")
    ax.set_ylabel("Eficiencia E(p) = S(p) / p")
    ax.set_xticks(range(1, p_max + 1))
    ax.set_ylim(0, max(1.05, e_max + 0.05))
    ax.set_title(_titulo("Eficiencia paralela", maquina, p_max))
    ax.legend()
    _guardar(fig, "g_eficiencia", maquina)


def plot_overhead(datos, maquina):
    """
    Item (h): overhead To(p) = p * T(p) - T(1).

    Mide el trabajo total agregado que introduce la paralelizacion. Fuentes
    esperadas en este esquema, para comentar en el informe:
      - creacion de los p procesos ('spawn' en Windows: re-importa el interprete
        y los modulos en cada hijo, mucho mas caro que 'fork');
      - serializacion y transferencia de X (~241 MB) hacia los workers, o el
        costo de escribir y mapear el archivo temporal si joblib usa memmap;
      - desbalance de carga: con B=48 y p=3, 48 no reparte parejo en todas las
        rondas;
      - contencion de memoria: los p procesos comparten un unico bus y cache L3,
        y este workload es intensivo en ancho de banda;
      - oversubscription de threads BLAS si no se limita t (ver item e).

    Panel izquierdo: To en segundos. Panel derecho: To / T(1), que permite
    comparar versiones con tiempos seriales muy distintos.
    """
    series = _series_barrido(datos, maquina)
    if not series:
        return
    p_max = _p_max(series)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for version, (p, med, _, _) in series.items():
        _, _, To = _metricas(p, med)
        T1 = med[p == 1][0]
        kw = dict(marker=MARCADORES[version], color=COLORES[version], label=ETIQUETAS[version])
        ax1.plot(p, To, **kw)
        ax2.plot(p, To / T1, **kw)
    for ax in (ax1, ax2):
        ax.axhline(0, color="gray", lw=1)
        ax.set_xlabel("Procesos p")
        ax.set_xticks(range(1, p_max + 1))
    ax1.set_ylabel("Overhead To(p) = p T(p) - T(1) [s]")
    ax2.set_ylabel("Overhead relativo To(p) / T(1)")
    ax1.legend(fontsize=9)
    fig.suptitle(_titulo("Overhead de la paralelizacion", maquina, p_max))
    _guardar(fig, "h_overhead", maquina)


def plot_grilla_pt(datos, maquina):
    """
    Item (i): tiempos para cada combinacion (p, t) con p * t <= cores logicos.

    Un heatmap con p en un eje y t en el otro es lo mas legible; dejar en blanco
    (o enmascaradas) las celdas que violan p * t <= p_max, y anotar el valor de
    tiempo dentro de cada celda para no obligar a leer la barra de color.
    """
    celdas = {(c[3], c[4]): v["mediana"] for c, v in datos.items()
              if c[0] == maquina and c[1] == "numpy" and c[4] >= 1}
    if not celdas:
        print("  (sin datos de --grilla: se omite plot_grilla_pt)")
        return
    p_max = max(max(p for p, _ in celdas), max(t for _, t in celdas))
    M = np.full((p_max, p_max), np.nan)
    for (p, t), v in celdas.items():
        M[p - 1, t - 1] = v
    mejor = min(celdas, key=celdas.get)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.ma.masked_invalid(M), origin="lower", cmap="viridis_r")
    for (p, t), v in celdas.items():
        negrita = (p, t) == mejor
        ax.text(t - 1, p - 1, f"{v:.2f} s" + ("\n(mejor)" if negrita else ""),
                ha="center", va="center", fontsize=10,
                fontweight="bold" if negrita else "normal",
                color="white" if v > np.nanmean(M) else "black")
    ax.set_xticks(range(p_max), range(1, p_max + 1))
    ax.set_yticks(range(p_max), range(1, p_max + 1))
    ax.set_xlabel("Threads BLAS por proceso t")
    ax.set_ylabel("Procesos p")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Tiempo de ejecucion [s]")
    ax.set_title(_titulo(f"Grilla (p, t) con p*t <= {p_max}, version numpy",
                         maquina, None))
    _guardar(fig, "i_grilla_pt", maquina)

    # Complemento: barras ordenadas por p*t, util si el heatmap queda muy vacio.
    orden = sorted(celdas, key=lambda c: (c[0] * c[1], c[0]))
    fig, ax = plt.subplots()
    colores = ["tab:red" if c == mejor else "tab:green" for c in orden]
    ax.bar([f"({p},{t})" for p, t in orden], [celdas[c] for c in orden], color=colores)
    ax.set_xlabel("Combinacion (p, t)")
    ax.set_ylabel("Tiempo de ejecucion [s]")
    ax.set_title(_titulo("Tiempos por combinacion (p, t)", maquina, p_max))
    _guardar(fig, "i_grilla_pt_barras", maquina)


def plot_variantes(datos, maquina):
    """
    Item (b): efecto de las mejoras sucesivas en bs_numpy (y bs_sklearn en p_max).

    Requiere haber corrido bench.py --variantes.
    """
    variantes = ["indices", "pesos_densos", "pesos"]
    series = {v: _serie(datos, maquina, "numpy", v) for v in variantes}
    series = {v: s for v, s in series.items() if s is not None}
    sk = {v: _serie(datos, maquina, "sklearn", v) for v in ("indices", "pesos")}
    sk = {v: s for v, s in sk.items() if s is not None}
    if len(series) < 2:
        print("  (sin datos de --variantes: se omite plot_variantes)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [3, 2]})
    for v, (p, med, lo, hi) in series.items():
        ax1.errorbar(p, med, yerr=[med - lo, hi - med], marker="o", capsize=3, label=v)
    ax1.set_xlabel("Procesos p")
    ax1.set_ylabel("Tiempo de ejecucion [s]")
    ax1.set_xticks(range(1, int(max(s[0].max() for s in series.values())) + 1))
    ax1.set_ylim(bottom=0)
    ax1.set_title("bs_numpy: variantes del ajuste")
    ax1.legend()

    if sk:
        nombres = list(sk)
        p_sk = int(max(s[0].max() for s in sk.values()))
        vals = [sk[v][1][sk[v][0] == p_sk][0] for v in nombres]
        ax2.bar(nombres, vals, color=["tab:gray", "tab:orange"][:len(nombres)])
        for i, val in enumerate(vals):
            ax2.text(i, val, f"{val:.1f} s", ha="center", va="bottom")
        ax2.set_ylabel("Tiempo de ejecucion [s]")
        ax2.set_title(f"bs_sklearn: variantes (p={p_sk})")
    else:
        ax2.set_visible(False)
    fig.suptitle(_titulo("Mejoras de implementacion (item b)", maquina, None))
    _guardar(fig, "b_variantes", maquina)


def plot_comparacion_maquinas(datos):
    """
    Item (j): mismo experimento en los dos computadores.

    Graficar T(p) de ambas maquinas en un panel y S(p) en otro. El punto del item
    es que las diferencias se justifiquen: numero de cores fisicos vs logicos
    (SMT no da speedup lineal), frecuencia base y turbo, throttling termico en
    notebooks, tamano de cache L3, ancho de banda de memoria, y sistema operativo
    (fork en Linux/macOS vs spawn en Windows cambia el costo de crear procesos).

    S(p) se normaliza por el T(1) de cada maquina para que la comparacion de
    ESCALABILIDAD no quede dominada por la diferencia de velocidad absoluta.
    """
    maquinas = sorted({c[0] for c in datos})
    if len(maquinas) < 2:
        print(f"  (el CSV solo tiene la maquina {maquinas}: se omite la comparacion)")
        return
    estilos = ["-", "--", ":", "-."]
    colores_maq = ["tab:purple", "tab:brown", "tab:cyan", "tab:olive"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    p_max = 1
    for i, maq in enumerate(maquinas):
        for j, (version, (p, med, _, _)) in enumerate(_series_barrido(datos, maq).items()):
            p_max = max(p_max, int(p.max()))
            S, _, _ = _metricas(p, med)
            kw = dict(color=colores_maq[i % 4], ls=estilos[j % 4],
                      marker=MARCADORES[version], label=f"{maq} - {version}")
            ax1.plot(p, med, **kw)
            ax2.plot(p, S, **kw)
    ideal = np.arange(1, p_max + 1)
    ax2.plot(ideal, ideal, color="gray", lw=1, label="Ideal S(p) = p")
    ax1.set_yscale("log")
    ax1.set_ylabel("Tiempo de ejecucion [s] (escala log)")
    ax2.set_ylabel("Speedup S(p) = T(1) / T(p)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Procesos p")
        ax.set_xticks(ideal)
    ax1.set_title("Tiempo de ejecucion T(p)")
    ax2.set_title("Speedup (T(1) de cada maquina y version)")
    ax2.legend(fontsize=8, loc="upper left")
    n = f"{common.N:_}".replace("_", ".")
    fig.suptitle(f"Comparacion entre computadores - N={n}, k={common.K}, B={common.B}")
    _guardar(fig, "j_comparacion", "maquinas")


# ---------------------------------------------------------------------------
# Tablas por consola
# ---------------------------------------------------------------------------

def imprimir_tablas(datos, maquina):
    """Tabla de T, S, E y To por version (items f, g, h) y de la grilla (item i)."""
    series = _series_barrido(datos, maquina)
    if series:
        print(f"\nTabla barrido - {maquina}")
        print(f"  {'version':8s} {'p':>2s} {'T(p) [s]':>9s} {'S(p)':>6s} {'E(p)':>6s} {'To(p) [s]':>10s}")
        for version, (p, med, _, _) in series.items():
            S, E, To = _metricas(p, med)
            for i in range(len(p)):
                print(f"  {version:8s} {p[i]:2d} {med[i]:9.2f} {S[i]:6.2f} {E[i]:6.2f} {To[i]:10.2f}")
    celdas = sorted((c[3], c[4], v["mediana"]) for c, v in datos.items()
                    if c[0] == maquina and c[1] == "numpy" and c[4] >= 1)
    if celdas:
        print(f"\nTabla grilla (p, t) - {maquina}")
        for p, t, v in celdas:
            print(f"  p={p} t={t}  T={v:.2f} s")


def main():
    """Lee el CSV, aplica el estilo comun y genera todas las figuras en figuras/."""
    parser = argparse.ArgumentParser(description="Figuras de la Tarea 1")
    parser.add_argument("--comparar", action="store_true",
                        help="item (j): figuras de todas las maquinas + comparacion")
    parser.add_argument("--maquina", default=None,
                        help="maquina del CSV a graficar (default: la actual)")
    parser.add_argument("--csv", default=common.CSV_TIEMPOS)
    args = parser.parse_args()

    plt.rcParams.update(ESTILO)
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    filas = common.leer_tiempos(args.csv)
    datos = agregar_tiempos(filas)
    presentes = sorted({f["maquina"] for f in filas})

    if args.comparar:
        maquinas = presentes
    elif args.maquina:
        maquinas = [args.maquina]
    else:
        actual = common.nombre_maquina()
        maquinas = [actual if actual in presentes else presentes[0]]

    for maq in maquinas:
        print(f"\nFiguras de {maq}:")
        plot_variantes(datos, maq)
        plot_tiempos(datos, maq)
        plot_speedup(datos, maq)
        plot_eficiencia(datos, maq)
        plot_overhead(datos, maq)
        plot_grilla_pt(datos, maq)
        imprimir_tablas(datos, maq)

    if args.comparar:
        print("\nComparacion entre maquinas:")
        plot_comparacion_maquinas(datos)


if __name__ == "__main__":
    main()
