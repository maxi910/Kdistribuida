"""
data_simulada.py

Generación de data simulada de flotación con:
- cinética de 1er orden con k distribuida (discreta) o
- modelo Klimpel o
- modelo Gamma como "modelo verdadero",
- discretización de tiempo fina (densa),
- error aleatorio en masa recolectada,
- error aleatorio en tiempo de muestreo.
"""

import numpy as np
import pandas as pd

from modelos_flotacion import (
    R_distribuida_discreta,
    R_klimpel,
    R_gamma,
)


def generar_data_simulada(
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    # descripción del modelo verdadero
    modelo_base="discreta",      # "discreta", "klimpel" o "gamma"
    params_base=None,            # dict con parámetros del modelo verdadero
    # ruido
    sigma_mass_rel=0.02,
    sigma_time=0.25,
    random_seed=42,
):
    """
    Genera data simulada de flotación.

    Parameters
    ----------
    t_max : float
        Tiempo máximo de simulación (s).
    dt : float
        Paso de tiempo (s) -> malla densa si es pequeño (ej. 0.1 s).
    M0 : float
        Masa alimentada inicial (por ejemplo, gramos).

    modelo_base : {"discreta", "klimpel", "gamma"}
        Modelo usado para generar la recuperación verdadera R_true(t).

    params_base : dict or None
        Parámetros del modelo verdadero:
        - si modelo_base == "discreta":
            {"k_values": [...],
             "w_values": [...],
             "R_inf": float}
        - si modelo_base == "klimpel":
            {"R_inf": float,
             "k_rect": float}
        - si modelo_base == "gamma":
            {"R_inf": float,
             "alpha": float,
             "theta": float}

        Si None, se usan valores por defecto razonables.

    sigma_mass_rel : float
        Desviación estándar relativa del error de masa (ej. 0.02 = 2%).
    sigma_time : float
        Desviación estándar del error en el tiempo (s).
    random_seed : int
        Semilla del generador aleatorio.

    Returns
    -------
    df : pandas.DataFrame
        Columnas:
        - t_true : tiempo verdadero (s)
        - t_obs  : tiempo observado con error (s)
        - R_true : recuperación verdadera (fracción)
        - R_obs  : recuperación observada (fracción)
        - M_true : masa verdadera recolectada
        - M_obs  : masa observada con error
    """
    rng = np.random.default_rng(random_seed)

    # Malla de tiempo verdadera (densa)
    t_true = np.arange(0.0, t_max + dt, dt)

    # --------------------------
    # 1) Definir modelo verdadero R_true(t)
    # --------------------------
    if params_base is None:
        params_base = {}

    modelo_base = modelo_base.lower()

    if modelo_base == "discreta":
        k_values = np.array(
            params_base.get("k_values", [0.02, 0.005, 0.001]),
            dtype=float,
        )
        w_values = np.array(
            params_base.get("w_values", [0.5, 0.3, 0.2]),
            dtype=float,
        )
        R_inf = float(params_base.get("R_inf", 1.0))

        R_true = R_distribuida_discreta(t_true, k_values, w_values, R_inf=R_inf)

    elif modelo_base == "klimpel":
        R_inf = float(params_base.get("R_inf", 0.9))
        k_rect = float(params_base.get("k_rect", 0.02))
        R_true = R_klimpel(t_true, R_inf, k_rect)

    elif modelo_base == "gamma":
        R_inf = float(params_base.get("R_inf", 0.9))
        alpha = float(params_base.get("alpha", 2.0))
        theta = float(params_base.get("theta", 50.0))
        R_true = R_gamma(t_true, R_inf, alpha, theta)

    else:
        raise ValueError(f"modelo_base no reconocido: {modelo_base}")

    # Masa verdadera
    M_true = M0 * R_true

    # --------------------------
    # 2) Ruido en tiempo de muestreo
    # --------------------------
    time_noise = rng.normal(loc=0.0, scale=sigma_time, size=t_true.size)
    t_obs = np.maximum(t_true + time_noise, 0.0)  # no permitir tiempos negativos

    # --------------------------
    # 3) Ruido en masa recolectada
    # --------------------------
    sigma_M = sigma_mass_rel * np.maximum(M_true, 1e-8)
    mass_noise = rng.normal(loc=0.0, scale=sigma_M)
    M_obs = np.maximum(M_true + mass_noise, 0.0)  # no permitir masas negativas

    # Recuperación observada a partir de la masa observada
    R_obs = M_obs / M0

    df = pd.DataFrame(
        {
            "t_true": t_true,
            "t_obs": t_obs,
            "R_true": R_true,
            "R_obs": R_obs,
            "M_true": M_true,
            "M_obs": M_obs,
        }
    )

    return df


if __name__ == "__main__":
    # Ejemplo rápido: modelo base discreto, por defecto
    df = generar_data_simulada()
    df.to_csv("flotacion_simulada_con_errores.csv", index=False)
    print(df.head())
    print("\nData simulada guardada en 'flotacion_simulada_con_errores.csv'")
