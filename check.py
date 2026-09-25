"""
Item (c): verificacion de correctitud y reproducibilidad.

Responde las tres preguntas del enunciado:
  1. Las tres versiones producen intervalos de confianza similares entre si y
     consistentes con los coeficientes verdaderos beta*.
  2. En que medida son equivalentes los resultados de las distintas versiones.
  3. Que condiciones se deben cumplir para que los resultados sean reproducibles
     entre ejecuciones.

No mide tiempos: corre cada version una sola vez con p fijo y compara los betas.

Uso:
    python check.py            # p=4 para las comparaciones
    python check.py -p 2
"""

import argparse
import itertools
import os

import numpy as np

import common
import bs_auto
import bs_sklearn
import bs_numpy


def cobertura(lo, hi, beta_true):
    """
    Fraccion de coeficientes cuyo intervalo [lo, hi] contiene al valor verdadero.

    Con un IC al 95% se espera una cobertura cercana a 0.95 sobre los k+1=301
    coeficientes. Un valor muy por debajo delata un error en el ajuste (por
    ejemplo, doble intercepto); uno de 1.00 exacto sugiere intervalos demasiado
    anchos.

    Con B=48 el IC por percentiles queda algo corto (los extremos de 48 valores
    subestiman las colas), asi que una cobertura de ~0.90-0.94 es lo esperable.
    """
    return float(np.mean((lo <= beta_true) & (beta_true <= hi)))


def comparar_intervalos(ic_a, ic_b):
    """
    Compara dos intervalos de confianza y retorna metricas de discrepancia.

    Responde la pregunta "en que medida son equivalentes". Las versiones NO van a
    dar bits identicos, porque cada una sortea sus indices con un generador
    distinto (BaggingRegressor usa random_state de sklearn, las otras usan
    default_rng). La equivalencia es estadistica, no numerica: los intervalos
    deben coincidir dentro del error de Monte Carlo propio de B=48 resamples.

    Retorna dict con diferencia absoluta maxima y media de los bordes.
    """
    lo_a, hi_a = ic_a
    lo_b, hi_b = ic_b
    dif = np.concatenate([np.abs(lo_a - lo_b), np.abs(hi_a - hi_b)])
    # Relativizar por el ancho: una diferencia de 0.002 no dice nada por si sola,
    # pero si el intervalo mide 0.01 es un 20% de su ancho.
    ancho = ((hi_a - lo_a) + (hi_b - lo_b)) / 2
    rel = dif / np.tile(ancho, 2)
    return {
        "max_abs": float(dif.max()),
        "media_abs": float(dif.mean()),
        "max_rel_ancho": float(rel.max()),
        "media_rel_ancho": float(rel.mean()),
        "ancho_medio_a": float(np.mean(hi_a - lo_a)),
        "ancho_medio_b": float(np.mean(hi_b - lo_b)),
    }


def verificar_reproducibilidad(p_lista=(1, 2, 4)):
    """
    Comprueba que una misma version entrega exactamente los mismos betas al
    variar p y al repetir la ejecucion.

    Esta es la respuesta a la pregunta 3 del item (c). Se cumple solo si:
      - cada resample deriva su semilla de forma deterministica de la semilla
        maestra (common.semillas_resamples), y no de un RNG compartido;
      - el resultado se reordena segun el indice de la tarea, sin depender del
        orden en que los workers terminen;
      - los datos de entrada son los mismos (de ahi el cache en disco).

    Nota: la reduccion en punto flotante dentro de BLAS puede variar con el
    numero de threads, asi que la igualdad puede ser exacta solo a nivel de
    indices sorteados y no bit a bit en los betas. Vale la pena documentarlo.
    """
    X, y, _ = common.cargar_datos()
    p_lista = [p for p in p_lista if p <= os.cpu_count()]

    print("\n== Reproducibilidad de bs_numpy (variante 'pesos') ==")
    ref = bs_numpy.run(X, y, p_lista[0])
    repetida = bs_numpy.run(X, y, p_lista[0])
    print(f"  misma p={p_lista[0]}, 2 ejecuciones:   array_equal={np.array_equal(ref, repetida)}"
          f"   allclose={np.allclose(ref, repetida)}")

    for p in p_lista[1:]:
        otra = bs_numpy.run(X, y, p)
        print(f"  p={p_lista[0]} vs p={p}:              array_equal={np.array_equal(ref, otra)}"
              f"   allclose={np.allclose(ref, otra)}"
              f"   max|dif|={np.abs(ref - otra).max():.1e}")

    # Mismo p, distinto numero de threads BLAS: cambia el orden de las sumas
    # dentro de X^T X, asi que aqui es donde puede romperse la igualdad bit a bit.
    for t in (1, os.cpu_count()):
        otra = bs_numpy.run(X, y, p_lista[0], threads=t)
        print(f"  p={p_lista[0]} sin limite vs t={t}:     array_equal={np.array_equal(ref, otra)}"
              f"   allclose={np.allclose(ref, otra)}"
              f"   max|dif|={np.abs(ref - otra).max():.1e}")

    # Semilla maestra distinta: otro resample, otro IC. Muestra que lo que
    # garantiza la reproducibilidad es fijar la semilla, no el codigo.
    otra = bs_numpy.run(X, y, p_lista[0], seed=common.SEED + 1)
    print(f"  semilla {common.SEED} vs {common.SEED + 1}:     "
          f"array_equal={np.array_equal(ref, otra)}   allclose={np.allclose(ref, otra)}")

    # Las variantes de bs_numpy y de bs_sklearn sortean los mismos indices con
    # la misma semilla, asi que deben coincidir hasta el redondeo.
    print("\n== Mismo resample, distinto algoritmo de ajuste ==")
    b = 8
    base = bs_numpy.run(X, y, p_lista[-1], b=b, variante="indices")
    for nombre, betas in [
        ("numpy/pesos_densos", bs_numpy.run(X, y, p_lista[-1], b=b, variante="pesos_densos")),
        ("numpy/pesos", bs_numpy.run(X, y, p_lista[-1], b=b, variante="pesos")),
        ("sklearn/indices", bs_sklearn.run(X, y, p_lista[-1], b=b, variante="indices")),
        ("sklearn/pesos", bs_sklearn.run(X, y, p_lista[-1], b=b, variante="pesos")),
    ]:
        print(f"  numpy/indices vs {nombre:19s} max|dif|={np.abs(base - betas).max():.1e}"
              f"   allclose={np.allclose(base, betas)}")


def main():
    """
    Corre las tres versiones, arma sus IC y emite la tabla comparativa del item (c).

    Salida esperada: por version, la cobertura respecto de beta*; luego la
    comparacion cruzada entre versiones; y por ultimo el chequeo de
    reproducibilidad.
    """
    parser = argparse.ArgumentParser(description="Item (c): correctitud y reproducibilidad")
    parser.add_argument("-p", type=int, default=os.cpu_count())
    args = parser.parse_args()

    X, y, beta_true = common.cargar_datos()

    # Paso 1: beta sobre el dataset completo, como referencia puntual.
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"Paso 1: max|beta_hat - beta*| = {np.abs(beta_hat - beta_true).max():.4f}")

    versiones = {"auto": bs_auto.run, "sklearn": bs_sklearn.run, "numpy": bs_numpy.run}
    ics = {}
    print(f"\n== Correctitud (p={args.p}, B={common.B}) ==")
    print(f"  {'version':8s} {'cobertura':>9s} {'ancho medio':>12s} "
          f"{'beta_hat en IC':>15s} {'max|media_b - beta_hat|':>24s}")
    for nombre, run in versiones.items():
        betas = run(X, y, args.p)
        lo, hi = common.intervalo_confianza(betas)
        ics[nombre] = (lo, hi)
        print(f"  {nombre:8s} {cobertura(lo, hi, beta_true):9.3f} "
              f"{np.mean(hi - lo):12.5f} {cobertura(lo, hi, beta_hat):15.3f} "
              f"{np.abs(betas.mean(axis=0) - beta_hat).max():24.5f}")

    print("\n== Equivalencia entre versiones (bordes del IC) ==")
    print(f"  {'par':18s} {'max|dif|':>9s} {'media|dif|':>11s} "
          f"{'max dif/ancho':>14s} {'media dif/ancho':>16s}")
    for a, b in itertools.combinations(ics, 2):
        m = comparar_intervalos(ics[a], ics[b])
        print(f"  {a + ' vs ' + b:18s} {m['max_abs']:9.5f} {m['media_abs']:11.5f} "
              f"{m['max_rel_ancho']:14.3f} {m['media_rel_ancho']:16.3f}")

    verificar_reproducibilidad()


if __name__ == "__main__":
    # Guard obligatorio en Windows: se lanzan procesos hijos via joblib.
    main()
