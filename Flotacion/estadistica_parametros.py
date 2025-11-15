"""
estadistica_parametros.py

Funciones para:
- Calcular errores estándar y intervalos de confianza de parámetros
- Calcular métricas de ajuste (SSE, RMSE, R², AIC, BIC)
"""

import numpy as np
from scipy.stats import t


def calcular_ic_parametros(popt, pcov, n_datos, alpha=0.95):
    """
    Calcula errores estándar e intervalos de confianza (IC) de los parámetros
    ajustados con curve_fit.

    Parameters
    ----------
    popt : array_like
        Parámetros óptimos (salida de curve_fit).
    pcov : array_like
        Matriz de covarianza de los parámetros (salida de curve_fit).
    n_datos : int
        Número de puntos de datos usados en el ajuste.
    alpha : float
        Nivel de confianza (ej. 0.95 para 95%).

    Returns
    -------
    resumen : dict
        Diccionario con:
        - 'popt'  : np.ndarray, parámetros
        - 'se'    : np.ndarray, errores estándar
        - 'ic_inf': np.ndarray, límite inferior IC
        - 'ic_sup': np.ndarray, límite superior IC
        - 'dof'   : int, grados de libertad
        - 't_crit': float, valor crítico t
    """
    popt = np.asarray(popt, dtype=float)
    pcov = np.asarray(pcov, dtype=float)

    n_params = popt.size
    dof = max(n_datos - n_params, 1)  # grados de libertad

    # Errores estándar = raíz de la diagonal de la covarianza
    se = np.sqrt(np.diag(pcov))

    # Valor crítico t para el nivel de confianza alpha
    t_crit = t.ppf(0.5 + alpha / 2.0, dof)

    ic_inf = popt - t_crit * se
    ic_sup = popt + t_crit * se

    resumen = {
        "popt": popt,
        "se": se,
        "ic_inf": ic_inf,
        "ic_sup": ic_sup,
        "dof": dof,
        "t_crit": t_crit,
    }
    return resumen


def metricas_ajuste(y_obs, y_pred, n_params):
    """
    Calcula métricas de ajuste a partir de observados y predichos.

    Parameters
    ----------
    y_obs : array_like
        Valores observados.
    y_pred : array_like
        Valores predichos por el modelo.
    n_params : int
        Número de parámetros del modelo.

    Returns
    -------
    met : dict
        Diccionario con:
        - 'SSE' : suma de cuadrados de los residuos
        - 'MSE' : error cuadrático medio (ajustado por grados de libertad)
        - 'RMSE': raíz del MSE
        - 'R2'  : coeficiente de determinación
        - 'AIC' : criterio de información de Akaike
        - 'BIC' : criterio de información bayesiano
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    resid = y_obs - y_pred
    n = y_obs.size

    # Suma de cuadrados de residuos
    sse = np.sum(resid ** 2)

    # MSE ajustado por grados de libertad
    dof = max(n - n_params, 1)
    mse = sse / dof
    rmse = np.sqrt(mse)

    # R²
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else np.nan

    # AIC y BIC (para comparar modelos)
    # Nota: usamos log(sse/n) como estimador de la varianza residual
    if sse <= 0:
        aic = np.nan
        bic = np.nan
    else:
        sigma2_hat = sse / n
        aic = n * np.log(sigma2_hat) + 2 * n_params
        bic = n * np.log(sigma2_hat) + n_params * np.log(n)

    met = {
        "SSE": sse,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "AIC": aic,
        "BIC": bic,
    }
    return met


def imprimir_resumen_parametros(nombres, resumen):
    """
    Imprime en pantalla un resumen bonito de parámetros con IC.

    Parameters
    ----------
    nombres : list of str
        Nombres de los parámetros (en el mismo orden que popt).
    resumen : dict
        Salida de calcular_ic_parametros().
    """
    popt = resumen["popt"]
    se = resumen["se"]
    ic_inf = resumen["ic_inf"]
    ic_sup = resumen["ic_sup"]
    dof = resumen["dof"]
    t_crit = resumen["t_crit"]

    print(f"Grados de libertad (dof) = {dof}")
    print(f"t crítico = {t_crit:.4f}\n")

    for name, val, err, lo, hi in zip(nombres, popt, se, ic_inf, ic_sup):
        print(
            f"{name:8s} = {val:10.5f}  ± {err:10.5f}  "
            f"[IC: {lo:10.5f} , {hi:10.5f}]"
        )


def imprimir_metricas(nombre_modelo, met):
    """
    Imprime métricas de ajuste en pantalla.

    Parameters
    ----------
    nombre_modelo : str
        Etiqueta del modelo (ej. 'Klimpel', 'Gamma').
    met : dict
        Salida de metricas_ajuste().
    """
    print(f"--- Métricas de ajuste: {nombre_modelo} ---")
    print(f"SSE  = {met['SSE']:.6f}")
    print(f"MSE  = {met['MSE']:.6f}")
    print(f"RMSE = {met['RMSE']:.6f}")
    print(f"R²   = {met['R2']:.6f}")
    print(f"AIC  = {met['AIC']:.3f}")
    print(f"BIC  = {met['BIC']:.3f}\n")


if __name__ == "__main__":
    # Pequeño test si ejecutas este archivo solo
    popt_test = np.array([1.0, 0.02])
    pcov_test = np.array([[1e-4, 0.0], [0.0, 1e-6]])
    resumen_test = calcular_ic_parametros(popt_test, pcov_test, n_datos=100)
    imprimir_resumen_parametros(["R_inf", "k"], resumen_test)
