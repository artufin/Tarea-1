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
    """
    # TODO: construir BaggingRegressor(estimator=LinearRegression(fit_intercept=False),
    #       n_estimators=b, n_jobs=p, bootstrap=True, random_state=seed)
    # TODO: .fit(X, y)
    # TODO: apilar los coeficientes de cada sub-modelo desde bag.estimators_
    raise NotImplementedError


def main():
    """Corrida individual de esta version: util para inspeccionarla sin el driver."""
    # TODO: cargar datos, correr run() para un p dado, imprimir el IC y el tiempo.
    pass


if __name__ == "__main__":
    # Guard obligatorio en Windows: el backend usa 'spawn', que re-importa este
    # modulo dentro de cada proceso hijo. Sin el guard, cada hijo volveria a
    # lanzar el experimento completo (recursion infinita de procesos).
    main()
