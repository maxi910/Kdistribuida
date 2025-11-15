"""
experimentos_modelo_true.py

Estudia el efecto del ruido cuando el modelo "verdadero" es exactamente:
- Klimpel, o
- Gamma.

Para cada nivel de ruido:
- genera data simulada con ese modelo base,
- ajusta el mismo tipo de modelo,
- calcula el sesgo de los parámetros y las métricas de ajuste.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_simulada import generar_data_simulada
from modelos_flotacion import (
    ajustar_klimpel,
    ajustar_gamma,
    R_klimpel,
    R_gamma,
)
from estadistica_parametros import metricas_ajuste


# -------------------------------------------------
# Experimentos para modelo verdadero Klimpel
# -------------------------------------------------

def experimento_klimpel_ruido_masa(
    sigma_mass_rel_values,
    params_true=None,
    sigma_time=0.25,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    if params_true is None:
        params_true = {"R_inf": 0.9, "k_rect": 0.02}

    R_inf_true = params_true["R_inf"]
    k_rect_true = params_true["k_rect"]

    filas = []

    for sigma_mass_rel in sigma_mass_rel_values:
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            modelo_base="klimpel",
            params_base=params_true,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        popt, _ = ajustar_klimpel(t_obs_ord, R_obs_ord)
        R_inf_est, k_rect_est = popt

        R_pred = R_klimpel(t_obs_ord, *popt)
        met = metricas_ajuste(R_obs_ord, R_pred, n_params=len(popt))

        filas.append(
            {
                "modelo_true": "Klimpel",
                "tipo_ruido": "masa",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "R_inf_true": R_inf_true,
                "k_rect_true": k_rect_true,
                "R_inf_est": R_inf_est,
                "k_rect_est": k_rect_est,
                "bias_R_inf": R_inf_est - R_inf_true,
                "bias_k_rect": k_rect_est - k_rect_true,
                "RMSE": met["RMSE"],
                "R2": met["R2"],
                "AIC": met["AIC"],
                "BIC": met["BIC"],
            }
        )

    return pd.DataFrame(filas)


def experimento_klimpel_ruido_tiempo(
    sigma_time_values,
    params_true=None,
    sigma_mass_rel=0.03,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    if params_true is None:
        params_true = {"R_inf": 0.9, "k_rect": 0.02}

    R_inf_true = params_true["R_inf"]
    k_rect_true = params_true["k_rect"]

    filas = []

    for sigma_time in sigma_time_values:
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            modelo_base="klimpel",
            params_base=params_true,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        popt, _ = ajustar_klimpel(t_obs_ord, R_obs_ord)
        R_inf_est, k_rect_est = popt

        R_pred = R_klimpel(t_obs_ord, *popt)
        met = metricas_ajuste(R_obs_ord, R_pred, n_params=len(popt))

        filas.append(
            {
                "modelo_true": "Klimpel",
                "tipo_ruido": "tiempo",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "R_inf_true": R_inf_true,
                "k_rect_true": k_rect_true,
                "R_inf_est": R_inf_est,
                "k_rect_est": k_rect_est,
                "bias_R_inf": R_inf_est - R_inf_true,
                "bias_k_rect": k_rect_est - k_rect_true,
                "RMSE": met["RMSE"],
                "R2": met["R2"],
                "AIC": met["AIC"],
                "BIC": met["BIC"],
            }
        )

    return pd.DataFrame(filas)


# -------------------------------------------------
# Experimentos para modelo verdadero Gamma
# -------------------------------------------------

def experimento_gamma_ruido_masa(
    sigma_mass_rel_values,
    params_true=None,
    sigma_time=0.25,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    if params_true is None:
        params_true = {"R_inf": 0.9, "alpha": 2.0, "theta": 50.0}

    R_inf_true = params_true["R_inf"]
    alpha_true = params_true["alpha"]
    theta_true = params_true["theta"]

    filas = []

    for sigma_mass_rel in sigma_mass_rel_values:
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            modelo_base="gamma",
            params_base=params_true,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        popt, _ = ajustar_gamma(t_obs_ord, R_obs_ord)
        R_inf_est, alpha_est, theta_est = popt

        R_pred = R_gamma(t_obs_ord, *popt)
        met = metricas_ajuste(R_obs_ord, R_pred, n_params=len(popt))

        filas.append(
            {
                "modelo_true": "Gamma",
                "tipo_ruido": "masa",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "R_inf_true": R_inf_true,
                "alpha_true": alpha_true,
                "theta_true": theta_true,
                "R_inf_est": R_inf_est,
                "alpha_est": alpha_est,
                "theta_est": theta_est,
                "bias_R_inf": R_inf_est - R_inf_true,
                "bias_alpha": alpha_est - alpha_true,
                "bias_theta": theta_est - theta_true,
                "RMSE": met["RMSE"],
                "R2": met["R2"],
                "AIC": met["AIC"],
                "BIC": met["BIC"],
            }
        )

    return pd.DataFrame(filas)


def experimento_gamma_ruido_tiempo(
    sigma_time_values,
    params_true=None,
    sigma_mass_rel=0.03,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    if params_true is None:
        params_true = {"R_inf": 0.9, "alpha": 2.0, "theta": 50.0}

    R_inf_true = params_true["R_inf"]
    alpha_true = params_true["alpha"]
    theta_true = params_true["theta"]

    filas = []

    for sigma_time in sigma_time_values:
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            modelo_base="gamma",
            params_base=params_true,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        popt, _ = ajustar_gamma(t_obs_ord, R_obs_ord)
        R_inf_est, alpha_est, theta_est = popt

        R_pred = R_gamma(t_obs_ord, *popt)
        met = metricas_ajuste(R_obs_ord, R_pred, n_params=len(popt))

        filas.append(
            {
                "modelo_true": "Gamma",
                "tipo_ruido": "tiempo",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "R_inf_true": R_inf_true,
                "alpha_true": alpha_true,
                "theta_true": theta_true,
                "R_inf_est": R_inf_est,
                "alpha_est": alpha_est,
                "theta_est": theta_est,
                "bias_R_inf": R_inf_est - R_inf_true,
                "bias_alpha": alpha_est - alpha_true,
                "bias_theta": theta_est - theta_true,
                "RMSE": met["RMSE"],
                "R2": met["R2"],
                "AIC": met["AIC"],
                "BIC": met["BIC"],
            }
        )

    return pd.DataFrame(filas)


# -------------------------------------------------
# Gráficos sencillos de bias y RMSE
# -------------------------------------------------

def plot_bias_y_rmse(df, x_col, modelo_true, titulo_suffix=""):
    """
    df: DataFrame de resultados para un modelo_true y un tipo de ruido.
    x_col: 'sigma_mass_rel' o 'sigma_time'
    """
    df = df.sort_values(x_col)

    # RMSE
    plt.figure()
    plt.plot(df[x_col], df["RMSE"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("RMSE en R(t)")
    plt.title(f"{modelo_true}: RMSE vs {x_col} {titulo_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Bias R_inf
    plt.figure()
    plt.plot(df[x_col], df["bias_R_inf"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel("Sesgo en R_inf (est - true)")
    plt.title(f"{modelo_true}: bias R_inf vs {x_col} {titulo_suffix}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    sigma_mass_rel_values = [0.0, 0.01, 0.03, 0.05, 0.10]
    sigma_time_values = [0.0, 0.05, 0.10, 0.25, 0.50]

    # ---- Modelo verdadero Klimpel ----
    df_K_masa = experimento_klimpel_ruido_masa(sigma_mass_rel_values)
    df_K_masa.to_csv("klimpel_true_ruido_masa.csv", index=False)
    print("=== Klimpel true, variando ruido en MASA ===")
    print(df_K_masa)
    plot_bias_y_rmse(df_K_masa, "sigma_mass_rel", "Klimpel", "(ruido masa)")

    df_K_tiempo = experimento_klimpel_ruido_tiempo(sigma_time_values)
    df_K_tiempo.to_csv("klimpel_true_ruido_tiempo.csv", index=False)
    print("\n=== Klimpel true, variando ruido en TIEMPO ===")
    print(df_K_tiempo)
    plot_bias_y_rmse(df_K_tiempo, "sigma_time", "Klimpel", "(ruido tiempo)")

    # ---- Modelo verdadero Gamma ----
    df_G_masa = experimento_gamma_ruido_masa(sigma_mass_rel_values)
    df_G_masa.to_csv("gamma_true_ruido_masa.csv", index=False)
    print("\n=== Gamma true, variando ruido en MASA ===")
    print(df_G_masa)
    plot_bias_y_rmse(df_G_masa, "sigma_mass_rel", "Gamma", "(ruido masa)")

    df_G_tiempo = experimento_gamma_ruido_tiempo(sigma_time_values)
    df_G_tiempo.to_csv("gamma_true_ruido_tiempo.csv", index=False)
    print("\n=== Gamma true, variando ruido en TIEMPO ===")
    print(df_G_tiempo)
    plot_bias_y_rmse(df_G_tiempo, "sigma_time", "Gamma", "(ruido tiempo)")
