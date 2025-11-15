"""
modelos_flotacion.py

Modelos cinéticos de flotación y funciones de ajuste:
- Primer orden clásico
- Primer orden con k distribuida (discreta)
- Modelo Klimpel
- Modelo Gamma
- Ajuste por mínimos cuadrados (curve_fit)
"""

import numpy as np
from scipy.optimize import curve_fit


# ============================
# Modelos R(t)
# ============================

def R_primer_orden(t, R_inf, k):
    """
    Modelo clásico de primer orden:
        R(t) = R_inf * (1 - exp(-k t))
    """
    t = np.asarray(t, dtype=float)
    return R_inf * (1.0 - np.exp(-k * t))


def R_distribuida_discreta(t, k_values, w_values, R_inf=1.0):
    """
    Modelo de primer orden con k distribuida (versión discreta):

        R(t) = R_inf * sum_i w_i * (1 - exp(-k_i t))

    Parameters
    ----------
    t : array_like
        Tiempos de flotación.
    k_values : array_like
        Valores de k_i para cada clase.
    w_values : array_like
        Pesos w_i (idealmente suman 1).
    R_inf : float
        Recuperación máxima (fracción).
    """
    t = np.asarray(t, dtype=float)
    k_values = np.asarray(k_values, dtype=float)
    w_values = np.asarray(w_values, dtype=float)

    # Normalizar pesos por seguridad
    w_values = w_values / np.sum(w_values)

    # Broadcasting: matriz (n_tiempos, n_clases)
    kt = np.outer(t, k_values)
    R = R_inf * np.sum(w_values * (1.0 - np.exp(-kt)), axis=1)
    return R


def R_klimpel(t, R_inf, k_rect):
    """
    Modelo Klimpel (distribución rectangular de constantes de velocidad):

        R(t) = R_inf * [1 - (1 / (k_rect * t)) * (1 - exp(-k_rect * t))]

    Notas
    -----
    - k_rect > 0
    - R_inf en [0,1] si trabajas con fracciones de recuperación.
    """
    t = np.asarray(t, dtype=float)
    R = np.zeros_like(t, dtype=float)

    # Evitar división por cero en t = 0
    mask = t > 0
    tt = t[mask]

    if k_rect <= 0:
        raise ValueError("k_rect debe ser positivo.")

    R[mask] = R_inf * (1.0 - (1.0 / (k_rect * tt)) * (1.0 - np.exp(-k_rect * tt)))
    # En t = 0, el límite es R(0) = 0 (ya está inicializado a cero)
    return R


def R_gamma(t, R_inf, alpha, theta):
    """
    Modelo Gamma (distribución Gamma de constantes de velocidad):

        R(t) = R_inf * [1 - (theta / (theta + t))**alpha]

    Parámetros
    ----------
    alpha > 0 : parámetro de forma.
    theta > 0 : escala (unidades de tiempo).
    """
    t = np.asarray(t, dtype=float)

    if alpha <= 0 or theta <= 0:
        raise ValueError("alpha y theta deben ser positivos.")

    return R_inf * (1.0 - (theta / (theta + t)) ** alpha)


# ============================
# Funciones de ajuste
# ============================

def ajustar_klimpel(t, R_obs, p0=(0.8, 0.02), bounds=(0, np.inf)):
    """
    Ajusta el modelo de Klimpel a datos de recuperación R(t).

    Parameters
    ----------
    t : array_like
        Tiempos de flotación.
    R_obs : array_like
        Recuperaciones observadas (fracción).
    p0 : tuple
        Valores iniciales (R_inf, k_rect).
    bounds : 2-tuple
        Límites para los parámetros (inferior, superior).

    Returns
    -------
    popt : np.ndarray
        Parámetros óptimos (R_inf, k_rect).
    pcov : np.ndarray
        Matriz de covarianza de los parámetros.
    """
    t = np.asarray(t, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)

    def _klimpel_for_fit(t_local, R_inf, k_rect):
        return R_klimpel(t_local, R_inf, k_rect)

    popt, pcov = curve_fit(
        _klimpel_for_fit,
        t,
        R_obs,
        p0=p0,
        bounds=bounds,
    )
    return popt, pcov


def ajustar_gamma(t, R_obs, p0=(0.8, 2.0, 50.0), bounds=(0, np.inf)):
    """
    Ajusta el modelo Gamma a datos de recuperación R(t).

    Parameters
    ----------
    t : array_like
        Tiempos de flotación.
    R_obs : array_like
        Recuperaciones observadas (fracción).
    p0 : tuple
        Valores iniciales (R_inf, alpha, theta).
    bounds : 2-tuple
        Límites para los parámetros (inferior, superior).

    Returns
    -------
    popt : np.ndarray
        Parámetros óptimos (R_inf, alpha, theta).
    pcov : np.ndarray
        Matriz de covarianza de los parámetros.
    """
    t = np.asarray(t, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)

    def _gamma_for_fit(t_local, R_inf, alpha, theta):
        return R_gamma(t_local, R_inf, alpha, theta)

    popt, pcov = curve_fit(
        _gamma_for_fit,
        t,
        R_obs,
        p0=p0,
        bounds=bounds,
    )
    return popt, pcov


if __name__ == "__main__":
    # Pequeño test rápido
    t_test = np.linspace(0, 300, 100)
    print("R_primer_orden(t=60) =", R_primer_orden(60.0, R_inf=0.9, k=0.02))
