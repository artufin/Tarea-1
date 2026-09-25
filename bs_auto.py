"""
Version 1 del item (b): bootstrap totalmente automatico.

Usa BaggingRegressor de sklearn con n_jobs=p. Tanto el resampleo como el
paralelismo son internos a sklearn: no escribimos el bucle sobre los B resamples
ni invocamos joblib directamente. Es la version de referencia (la "caja negra")
contra la que se comparan bs_sklearn.py y bs_numpy.py.
"""

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

import common


def run(X, y, p, b=common.B, seed=common.SEED):
    """
    Ejecuta los b resamples con BaggingRegressor y retorna los coeficientes.

    Retorna un array (b, k+1): una fila por resample.

    Detalles que importan para que sea comparable con las otras dos versiones:
      - fit_intercept=False, porque X YA trae la columna de unos; dejarlo en True
        agregaria un segundo intercepto y los betas no serian comparables.
      - bootstrap=True y max_samples=1.0 son los defaults y corresponden
        exactamente al Paso 2(i): sortear N indices con reemplazo.
      - random_state=seed para la reproducibilidad del item (c).

    Nota para el item (b): como LinearRegression acepta sample_weight,
    BaggingRegressor NO materializa X[idx]; convierte los indices sorteados en
    conteos (np.bincount) y los pasa como pesos. Es el mismo truco que la
    variante "pesos" de bs_numpy.py, pero el ajuste sigue siendo lstsq (SVD),
    mas caro que resolver el sistema normal.

    Con max_features=1.0 y bootstrap_features=False no se indexan columnas, asi
    que cada coef_ ya viene en el orden original de X.
    """
    bag = BaggingRegressor(
        estimator=LinearRegression(fit_intercept=False),
        n_estimators=b,
        n_jobs=p,
        bootstrap=True,
        random_state=seed,
    )
    bag.fit(X, y)
    return np.vstack([est.coef_ for est in bag.estimators_])


def main():
    """Corrida individual de esta version: util para inspeccionarla sin el driver."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-p", type=int, default=1, help="numero de procesos (n_jobs)")
    args = parser.parse_args()

    X, y, beta_true = common.cargar_datos()
    t0 = time.perf_counter()
    betas = run(X, y, args.p)
    tiempo = time.perf_counter() - t0

    lo, hi = common.intervalo_confianza(betas)
    cobertura = np.mean((lo <= beta_true) & (beta_true <= hi))
    print(f"bs_auto  p={args.p}  tiempo={tiempo:.2f} s  cobertura IC95={cobertura:.3f}")
    for j in range(3):
        print(f"  beta_{j}: IC=[{lo[j]:+.4f}, {hi[j]:+.4f}]  verdadero={beta_true[j]:+.4f}")


if __name__ == "__main__":
    # Guard obligatorio en Windows: el backend usa 'spawn', que re-importa este
    # modulo dentro de cada proceso hijo. Sin el guard, cada hijo volveria a
    # lanzar el experimento completo (recursion infinita de procesos).
    main()
