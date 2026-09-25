"""
Driver de medicion: ejecuta los experimentos y deja los tiempos en un CSV.

Cubre el item (f) --- barrido de p sobre las tres versiones --- y el item (i)
--- grilla combinada de procesos p y threads internos t. No grafica nada: eso
queda en plots.py, para poder re-graficar sin volver a medir (cada barrido
completo toma varios minutos).

Uso tipico:
    python bench.py --barrido
    python bench.py --grilla
    python bench.py --variantes

Convencion de la columna 't' del CSV: t=0 significa "sin threadpool_limits",
es decir, el default del backend (loky limita solo a cpu_count // p).
"""

import argparse
import os
import time

import common
import bs_auto
import bs_sklearn
import bs_numpy

# Registro de las tres versiones del item (b). La clave es el nombre que queda
# escrito en el CSV y que plots.py usa en las leyendas. Cada version corre con
# su mejor variante (ver --variantes).
VERSIONES = {
    "auto": bs_auto.run,
    "sklearn": bs_sklearn.run,
    "numpy": bs_numpy.run,
}

# Variante con la que corre cada version en el barrido principal; es la que
# queda escrita en la columna 'variante' del CSV.
VARIANTE_POR_VERSION = {
    "auto": "bagging",
    "sklearn": "pesos",
    "numpy": "pesos",
}

# Version y variante mas eficientes del item (b): las usa la grilla del item (i).
MEJOR_VERSION = ("numpy", "pesos")


def _medir(etiqueta, fn, warmup, repeticiones, registrar):
    """
    Cronometra fn() y registra una fila por repeticion.

    El warmup NO es una corrida completa: basta con ejecutar p resamples para
    levantar el pool de workers de joblib e importar NumPy/sklearn en cada uno,
    que son los costos unicos que se quieren excluir de la medicion.
    """
    warmup()
    tiempos = common.cronometrar(fn, repeticiones=repeticiones, descartar_warmup=False)
    for i, t in enumerate(tiempos):
        registrar(i, t)
    mediana = sorted(tiempos)[len(tiempos) // 2]
    print(f"  {etiqueta:40s} mediana={mediana:8.2f} s   "
          f"[{', '.join(f'{t:.2f}' for t in tiempos)}]", flush=True)


def barrido_p(versiones=None, p_max=None, repeticiones=3):
    """
    Item (f): mide T(p) para p en {1, ..., p_max} en cada version.

    p_max es el numero de cores logicos (os.cpu_count()); en este equipo, 4.
    Para cada (version, p) se cronometran varias repeticiones y se escribe una
    fila por repeticion, de modo que plots.py pueda tomar la mediana y mostrar
    la dispersion.

    Los datos se cargan UNA sola vez, fuera del cronometro: el enunciado mide el
    costo del bootstrap paralelo, no el de generar la matriz.
    """
    versiones = versiones or list(VERSIONES)
    p_max = p_max or os.cpu_count()
    X, y, _ = common.cargar_datos()

    print(f"\n== Barrido de p (item f): p = 1..{p_max}, {repeticiones} repeticiones ==")
    for version in versiones:
        run = VERSIONES[version]
        variante = VARIANTE_POR_VERSION[version]
        for p in range(1, p_max + 1):
            _medir(
                f"{version:8s} ({variante}) p={p}",
                fn=lambda: run(X, y, p),
                warmup=lambda: run(X, y, p, b=p),
                repeticiones=repeticiones,
                registrar=lambda i, t: common.registrar_tiempo(
                    version, variante, p, 0, i, t),
            )


def grilla_pt(p_max=None, repeticiones=3):
    """
    Item (i): mide la version mas eficiente variando p y t simultaneamente.

    Solo se evaluan combinaciones con p * t <= p_max, para no sobresuscribir la
    maquina. Con p_max=4 las combinaciones validas son:
        (1,1) (1,2) (1,3) (1,4) (2,1) (2,2) (3,1) (4,1)

    La hipotesis a contrastar es que el optimo NO esta en t=1 ni en p=1, sino en
    algun punto intermedio, porque cada resample es una operacion BLAS grande
    (X^T X con k+1=301) que si se beneficia de multi-threading.
    """
    p_max = p_max or os.cpu_count()
    version, variante = MEJOR_VERSION
    X, y, _ = common.cargar_datos()
    pares = [(p, t) for p in range(1, p_max + 1)
             for t in range(1, p_max + 1) if p * t <= p_max]

    print(f"\n== Grilla (p, t) con p*t <= {p_max} (item i): {version}/{variante} ==")
    for p, t in pares:
        _medir(
            f"{version} ({variante}) p={p} t={t}",
            fn=lambda: bs_numpy.run(X, y, p, variante=variante, threads=t),
            warmup=lambda: bs_numpy.run(X, y, p, b=p, variante=variante, threads=t),
            repeticiones=repeticiones,
            registrar=lambda i, tiempo: common.registrar_tiempo(
                version, variante, p, t, i, tiempo),
        )


def comparar_variantes(repeticiones=3, p_max=None):
    """
    Apoyo al item (b): mide las variantes de cada version para cuantificar la mejora.

    Es la evidencia concreta de la frase del enunciado "es muy probable que su
    primera implementacion no sea la mas eficiente".

    - bs_numpy: 'indices' vs 'pesos_densos' vs 'pesos', en p = 1..p_max.
    - bs_sklearn: 'indices' vs 'pesos', solo en p = p_max, porque la variante
      'indices' tarda ~3 s por resample y barrer todo p tomaria media hora.
    """
    p_max = p_max or os.cpu_count()
    X, y, _ = common.cargar_datos()

    print(f"\n== Variantes (item b), {repeticiones} repeticiones ==")
    for variante in bs_numpy.VARIANTES:
        for p in range(1, p_max + 1):
            _medir(
                f"numpy ({variante}) p={p}",
                fn=lambda: bs_numpy.run(X, y, p, variante=variante),
                warmup=lambda: bs_numpy.run(X, y, p, b=p, variante=variante),
                repeticiones=repeticiones,
                registrar=lambda i, t: common.registrar_tiempo(
                    "numpy", variante, p, 0, i, t),
            )
    for variante in bs_sklearn.VARIANTES:
        p = p_max
        _medir(
            f"sklearn ({variante}) p={p}",
            fn=lambda: bs_sklearn.run(X, y, p, variante=variante),
            warmup=lambda: bs_sklearn.run(X, y, p, b=p, variante=variante),
            repeticiones=repeticiones,
            registrar=lambda i, t: common.registrar_tiempo(
                "sklearn", variante, p, 0, i, t),
        )


def main():
    """CLI del driver: --barrido (item f), --grilla (item i), --variantes (item b)."""
    parser = argparse.ArgumentParser(description="Mediciones de tiempo de la Tarea 1")
    parser.add_argument("--barrido", action="store_true", help="item (f): 3 versiones x p")
    parser.add_argument("--grilla", action="store_true", help="item (i): grilla (p, t)")
    parser.add_argument("--variantes", action="store_true", help="item (b): mejoras")
    parser.add_argument("--versiones", nargs="+", choices=list(VERSIONES),
                        help="restringe el barrido a estas versiones")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--p-max", type=int, default=None,
                        help="default: numero de cores logicos")
    args = parser.parse_args()

    if not (args.barrido or args.grilla or args.variantes):
        parser.error("indicar al menos uno de --barrido, --grilla, --variantes")

    print("Ficha de la maquina:")
    for clave, valor in common.info_maquina().items():
        print(f"  {clave:14s} {valor}")
    print(f"Resultados -> {common.CSV_TIEMPOS}")

    t0 = time.perf_counter()
    if args.variantes:
        comparar_variantes(args.repeticiones, args.p_max)
    if args.barrido:
        barrido_p(args.versiones, args.p_max, args.repeticiones)
    if args.grilla:
        grilla_pt(args.p_max, args.repeticiones)
    print(f"\nTiempo total de medicion: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    # Guard obligatorio en Windows: este modulo lanza procesos hijos via joblib.
    main()
