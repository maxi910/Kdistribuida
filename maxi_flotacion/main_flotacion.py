
import numpy as np
import matplotlib.pyplot as plt

from data_simulada import generar_data_simulada
from modelos_flotacion import (
    R_klimpel,
    R_gamma,
    ajustar_klimpel,
    ajustar_gamma,
)
from estadistica_parametros import (
    calcular_ic_parametros,
    metricas_ajuste,
    imprimir_resumen_parametros,
    imprimir_metricas,
)


def main():
    M0 = 100.0

    # 1. Generar data simulada
    df = generar_data_simulada(
        t_max=300.0,
        dt=0.1,
        M0=M0,
        modelo_base="gamma",  
        params_base={
            "R_inf": 0.78,
            "alpha": 0.73,
            "theta": 6.66,
        },
        sigma_mass_rel=0.01,
        sigma_time=0.25,
        random_seed=546,
    )

    t_true = df["t_true"].values
    R_true = df["R_true"].values
    t_obs = df["t_obs"].values
    R_obs = df["R_obs"].values

    orden = np.argsort(t_obs)
    t_obs_ord = t_obs[orden]
    R_obs_ord = R_obs[orden]

    n_datos = t_obs_ord.size

    # 2. Ajuste de Klimpel y Gamma

    popt_klimpel, pcov_klimpel = ajustar_klimpel(t_obs_ord, R_obs_ord)
    popt_gamma, pcov_gamma = ajustar_gamma(t_obs_ord, R_obs_ord)

    print("=== Parámetros Klimpel ajustados (sin IC) ===")
    print(f"R_inf  = {popt_klimpel[0]:.5f}")
    print(f"k_rect = {popt_klimpel[1]:.5f}  [1/s]\n")

    print("=== Parámetros Gamma ajustados (sin IC) ===")
    print(f"R_inf  = {popt_gamma[0]:.5f}")
    print(f"alpha  = {popt_gamma[1]:.5f}")
    print(f"theta  = {popt_gamma[2]:.5f}  [s]\n")

    # 3. IC de parámetros

    resumen_K = calcular_ic_parametros(popt_klimpel, pcov_klimpel, n_datos, alpha=0.95)
    resumen_G = calcular_ic_parametros(popt_gamma, pcov_gamma, n_datos, alpha=0.95)

    print("=== Intervalos de confianza (95%) - Klimpel ===")
    imprimir_resumen_parametros(["R_inf", "k_rect"], resumen_K)
    print()

    print("=== Intervalos de confianza (95%) - Gamma ===")
    imprimir_resumen_parametros(["R_inf", "alpha", "theta"], resumen_G)
    print()

    # 4. Métricas de ajuste

    R_pred_K = R_klimpel(t_obs_ord, *popt_klimpel)
    R_pred_G = R_gamma(t_obs_ord, *popt_gamma)

    met_K = metricas_ajuste(R_obs_ord, R_pred_K, n_params=len(popt_klimpel))
    met_G = metricas_ajuste(R_obs_ord, R_pred_G, n_params=len(popt_gamma))

    imprimir_metricas("Klimpel", met_K)
    imprimir_metricas("Gamma", met_G)

    # ============================
    # 5. Gráfico comparativo
    # ============================
    t_plot = np.linspace(0.0, t_true.max(), 400)
    R_fit_K = R_klimpel(t_plot, *popt_klimpel)
    R_fit_G = R_gamma(t_plot, *popt_gamma)

    plt.figure()
    plt.scatter(
        t_obs_ord,
        R_obs_ord,
        s=10,
        alpha=0.7,
        label="Datos simulados (ruido)",
    )
    plt.plot(
        t_true,
        R_true,
        "--",
        label="Curva verdadera (Gamma)",
    )
    plt.plot(t_plot, R_fit_K, label="Ajuste Klimpel")
    plt.plot(t_plot, R_fit_G, label="Ajuste Gamma")
    plt.xlabel("Tiempo de flotación [s]")
    plt.ylabel("Recuperación R(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
