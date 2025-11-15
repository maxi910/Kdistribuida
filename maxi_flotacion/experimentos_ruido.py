"""
experimentos_ruido.py

Explora el efecto de distintos niveles de ruido en:
- masa recolectada (sigma_mass_rel)
- tiempo de muestreo (sigma_time)

Para cada caso:
- genera data simulada,
- ajusta modelos de Klimpel y Gamma,
- calcula métricas de ajuste (SSE, RMSE, R², AIC, BIC),
- guarda resultados en CSV,
- grafica RMSE y parámetros vs nivel de ruido.
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


# ============================================================
# Experimentos de barrido de ruido
# ============================================================

def experimento_ruido_masa(
    sigma_mass_rel_values,
    sigma_time=0.25,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    """
    Barre distintos niveles de ruido en masa (sigma_mass_rel),
    manteniendo fijo el ruido en tiempo (sigma_time).
    """
    resultados = []

    for sigma_mass_rel in sigma_mass_rel_values:
        # Generar data simulada
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        # Ordenar por tiempo observado
        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        # Ajuste de Klimpel y Gamma
        popt_K, _ = ajustar_klimpel(t_obs_ord, R_obs_ord)
        popt_G, _ = ajustar_gamma(t_obs_ord, R_obs_ord)

        R_pred_K = R_klimpel(t_obs_ord, *popt_K)
        R_pred_G = R_gamma(t_obs_ord, *popt_G)

        met_K = metricas_ajuste(R_obs_ord, R_pred_K, n_params=len(popt_K))
        met_G = metricas_ajuste(R_obs_ord, R_pred_G, n_params=len(popt_G))

        # Guardar resultados Klimpel
        resultados.append(
            {
                "tipo_ruido": "masa",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "modelo": "Klimpel",
                "R_inf": popt_K[0],
                "k_rect_alpha": popt_K[1],  # aquí es k_rect
                "theta": np.nan,
                "RMSE": met_K["RMSE"],
                "R2": met_K["R2"],
                "AIC": met_K["AIC"],
                "BIC": met_K["BIC"],
            }
        )

        # Guardar resultados Gamma
        resultados.append(
            {
                "tipo_ruido": "masa",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "modelo": "Gamma",
                "R_inf": popt_G[0],
                "k_rect_alpha": popt_G[1],  # aquí es alpha
                "theta": popt_G[2],
                "RMSE": met_G["RMSE"],
                "R2": met_G["R2"],
                "AIC": met_G["AIC"],
                "BIC": met_G["BIC"],
            }
        )

    return pd.DataFrame(resultados)


def experimento_ruido_tiempo(
    sigma_time_values,
    sigma_mass_rel=0.03,
    t_max=300.0,
    dt=0.1,
    M0=100.0,
    random_seed=123,
):
    """
    Barre distintos niveles de ruido en tiempo (sigma_time),
    manteniendo fijo el ruido en masa (sigma_mass_rel).
    """
    resultados = []

    for sigma_time in sigma_time_values:
        # Generar data simulada
        df = generar_data_simulada(
            t_max=t_max,
            dt=dt,
            M0=M0,
            sigma_mass_rel=sigma_mass_rel,
            sigma_time=sigma_time,
            random_seed=random_seed,
        )

        t_obs = df["t_obs"].values
        R_obs = df["R_obs"].values

        # Ordenar por tiempo observado
        orden = np.argsort(t_obs)
        t_obs_ord = t_obs[orden]
        R_obs_ord = R_obs[orden]

        # Ajuste de Klimpel y Gamma
        popt_K, _ = ajustar_klimpel(t_obs_ord, R_obs_ord)
        popt_G, _ = ajustar_gamma(t_obs_ord, R_obs_ord)

        R_pred_K = R_klimpel(t_obs_ord, *popt_K)
        R_pred_G = R_gamma(t_obs_ord, *popt_G)

        met_K = metricas_ajuste(R_obs_ord, R_pred_K, n_params=len(popt_K))
        met_G = metricas_ajuste(R_obs_ord, R_pred_G, n_params=len(popt_G))

        resultados.append(
            {
                "tipo_ruido": "tiempo",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "modelo": "Klimpel",
                "R_inf": popt_K[0],
                "k_rect_alpha": popt_K[1],
                "theta": np.nan,
                "RMSE": met_K["RMSE"],
                "R2": met_K["R2"],
                "AIC": met_K["AIC"],
                "BIC": met_K["BIC"],
            }
        )

        resultados.append(
            {
                "tipo_ruido": "tiempo",
                "sigma_mass_rel": sigma_mass_rel,
                "sigma_time": sigma_time,
                "modelo": "Gamma",
                "R_inf": popt_G[0],
                "k_rect_alpha": popt_G[1],  # alpha
                "theta": popt_G[2],
                "RMSE": met_G["RMSE"],
                "R2": met_G["R2"],
                "AIC": met_G["AIC"],
                "BIC": met_G["BIC"],
            }
        )

    return pd.DataFrame(resultados)


# ============================================================
# Funciones de gráficos
# ============================================================

def plot_ruido_masa(df_masa):
    """Gráficos: RMSE y parámetros vs sigma_mass_rel."""
    dfK = df_masa[df_masa["modelo"] == "Klimpel"].sort_values("sigma_mass_rel")
    dfG = df_masa[df_masa["modelo"] == "Gamma"].sort_values("sigma_mass_rel")

    # --- RMSE vs ruido en masa ---
    plt.figure()
    plt.plot(dfK["sigma_mass_rel"], dfK["RMSE"], marker="o", label="Klimpel")
    plt.plot(dfG["sigma_mass_rel"], dfG["RMSE"], marker="s", label="Gamma")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("RMSE en R(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- R_inf vs ruido en masa ---
    plt.figure()
    plt.plot(dfK["sigma_mass_rel"], dfK["R_inf"], marker="o", label="Klimpel")
    plt.plot(dfG["sigma_mass_rel"], dfG["R_inf"], marker="s", label="Gamma")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("R_inf estimado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- k_rect (Klimpel) vs ruido en masa ---
    plt.figure()
    plt.plot(dfK["sigma_mass_rel"], dfK["k_rect_alpha"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("k_rect (Klimpel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- alpha y theta (Gamma) vs ruido en masa ---
    plt.figure()
    plt.plot(dfG["sigma_mass_rel"], dfG["k_rect_alpha"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("alpha (Gamma)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(dfG["sigma_mass_rel"], dfG["theta"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("theta (Gamma) [s]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_ruido_tiempo(df_tiempo):
    """Gráficos: RMSE y parámetros vs sigma_time."""
    dfK = df_tiempo[df_tiempo["modelo"] == "Klimpel"].sort_values("sigma_time")
    dfG = df_tiempo[df_tiempo["modelo"] == "Gamma"].sort_values("sigma_time")

    # --- RMSE vs ruido en tiempo ---
    plt.figure()
    plt.plot(dfK["sigma_time"], dfK["RMSE"], marker="o", label="Klimpel")
    plt.plot(dfG["sigma_time"], dfG["RMSE"], marker="s", label="Gamma")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("RMSE en R(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- R_inf vs ruido en tiempo ---
    plt.figure()
    plt.plot(dfK["sigma_time"], dfK["R_inf"], marker="o", label="Klimpel")
    plt.plot(dfG["sigma_time"], dfG["R_inf"], marker="s", label="Gamma")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("R_inf estimado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- k_rect (Klimpel) vs ruido en tiempo ---
    plt.figure()
    plt.plot(dfK["sigma_time"], dfK["k_rect_alpha"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("k_rect (Klimpel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- alpha y theta (Gamma) vs ruido en tiempo ---
    plt.figure()
    plt.plot(dfG["sigma_time"], dfG["k_rect_alpha"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("alpha (Gamma)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(dfG["sigma_time"], dfG["theta"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("theta (Gamma) [s]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Bloque principal
# ============================================================

if __name__ == "__main__":
    # Rangos de ruido a probar
    sigma_mass_rel_values = [0.0, 0.01, 0.03, 0.05, 0.10]  # 0% a 10% relativo
    sigma_time_values = [0.0, 0.05, 0.10, 0.25, 0.50]      # en segundos

    # ---- Experimento variando ruido en masa ----
    df_masa = experimento_ruido_masa(
        sigma_mass_rel_values=sigma_mass_rel_values,
        sigma_time=0.25,
    )
    print("=== Resultados variando ruido en MASA ===")
    print(df_masa)
    df_masa.to_csv("resultados_ruido_masa.csv", index=False)

    # Gráficos para ruido en masa
    plot_ruido_masa(df_masa)

    # ---- Experimento variando ruido en tiempo ----
    df_tiempo = experimento_ruido_tiempo(
        sigma_time_values=sigma_time_values,
        sigma_mass_rel=0.03,
    )
    print("\n=== Resultados variando ruido en TIEMPO ===")
    print(df_tiempo)
    df_tiempo.to_csv("resultados_ruido_tiempo.csv", index=False)

    # Gráficos para ruido en tiempo
    plot_ruido_tiempo(df_tiempo)
