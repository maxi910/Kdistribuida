import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1) KLIMPEL VERDADERO – RUIDO EN MASA
#    lee klimpel_true_ruido_masa.csv
# ============================================================

def plot_klimpel_ruido_masa(csv_path="klimpel_true_ruido_masa.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("sigma_mass_rel")

    # RMSE vs ruido en masa
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["RMSE"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("RMSE en R(t)")
    plt.title("Klimpel verdadero: RMSE vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en R_inf vs ruido en masa
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["bias_R_inf"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("Sesgo en R_inf (estimado - verdadero)")
    plt.title("Klimpel verdadero: bias R_inf vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en k_rect vs ruido en masa
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["bias_k_rect"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("Sesgo en k_rect (estimado - verdadero)")
    plt.title("Klimpel verdadero: bias k_rect vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2) KLIMPEL VERDADERO – RUIDO EN TIEMPO
#    lee klimpel_true_ruido_tiempo.csv
# ============================================================

def plot_klimpel_ruido_tiempo(csv_path="klimpel_true_ruido_tiempo.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("sigma_time")

    # RMSE vs ruido en tiempo
    plt.figure()
    plt.plot(df["sigma_time"], df["RMSE"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("RMSE en R(t)")
    plt.title("Klimpel verdadero: RMSE vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en R_inf vs ruido en tiempo
    plt.figure()
    plt.plot(df["sigma_time"], df["bias_R_inf"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("Sesgo en R_inf (estimado - verdadero)")
    plt.title("Klimpel verdadero: bias R_inf vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en k_rect vs ruido en tiempo
    plt.figure()
    plt.plot(df["sigma_time"], df["bias_k_rect"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("Sesgo en k_rect (estimado - verdadero)")
    plt.title("Klimpel verdadero: bias k_rect vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) GAMMA VERDADERO – RUIDO EN MASA
#    lee gamma_true_ruido_masa.csv
# ============================================================

def plot_gamma_ruido_masa(csv_path="gamma_true_ruido_masa.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("sigma_mass_rel")

    # RMSE vs ruido en masa
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["RMSE"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("RMSE en R(t)")
    plt.title("Gamma verdadero: RMSE vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en R_inf
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["bias_R_inf"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("Sesgo en R_inf (estimado - verdadero)")
    plt.title("Gamma verdadero: bias R_inf vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en alpha
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["bias_alpha"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("Sesgo en alpha (estimado - verdadero)")
    plt.title("Gamma verdadero: bias alpha vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en theta
    plt.figure()
    plt.plot(df["sigma_mass_rel"], df["bias_theta"], marker="o")
    plt.xlabel("sigma_mass_rel (ruido relativo en masa)")
    plt.ylabel("Sesgo en theta (estimado - verdadero) [s]")
    plt.title("Gamma verdadero: bias theta vs ruido en masa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) GAMMA VERDADERO – RUIDO EN TIEMPO
#    lee gamma_true_ruido_tiempo.csv
# ============================================================

def plot_gamma_ruido_tiempo(csv_path="gamma_true_ruido_tiempo.csv"):
    df = pd.read_csv(csv_path)
    df = df.sort_values("sigma_time")

    # RMSE vs ruido en tiempo
    plt.figure()
    plt.plot(df["sigma_time"], df["RMSE"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("RMSE en R(t)")
    plt.title("Gamma verdadero: RMSE vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en R_inf
    plt.figure()
    plt.plot(df["sigma_time"], df["bias_R_inf"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("Sesgo en R_inf (estimado - verdadero)")
    plt.title("Gamma verdadero: bias R_inf vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en alpha
    plt.figure()
    plt.plot(df["sigma_time"], df["bias_alpha"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("Sesgo en alpha (estimado - verdadero)")
    plt.title("Gamma verdadero: bias alpha vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Sesgo en theta
    plt.figure()
    plt.plot(df["sigma_time"], df["bias_theta"], marker="o")
    plt.xlabel("sigma_time (ruido en tiempo) [s]")
    plt.ylabel("Sesgo en theta (estimado - verdadero) [s]")
    plt.title("Gamma verdadero: bias theta vs ruido en tiempo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Comenta/descomenta lo que quieras ver
    plot_klimpel_ruido_masa()
    plot_klimpel_ruido_tiempo()
    plot_gamma_ruido_masa()
    plot_gamma_ruido_tiempo()
