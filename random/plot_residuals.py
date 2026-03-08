import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from repitframework.plot_utils import load_metrics
from repitframework.Metrics.ResidualNaturalConvection import residual_mass, residual_momentum, residual_heat

GROUND_TRUTH_DIR = "/home/shilaj/shilaj_data/repitframework/repitframework/Assets/natural_convection_case1"
PREDICTION_DIR  = "/data/disk3/shilaj_data/repit_backups/repit_case_study/final_paper/natural_convection_case1_study/10000timesteps/2epochs5res/natural_convection_case1"
METRICS_DIR = "/data/disk3/shilaj_data/repit_backups/repit_case_study/final_paper/natural_convection_case1_study/10000timesteps/2epochs5res/prediction_metrics.ndjson"

# --- Palette (Nature style extended for 3 lines) ---
COL_BLUE_DARK = "#007CD3"   # Mass
COL_ORANGE_DARK = "#D5A000" # Momentum
COL_GREEN_DARK = "#009E73"  # Heat

def moving_average(a, n=10):
    """Simple moving average for smoothing noisy residuals."""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def chunk_time_series():
    metrics = load_metrics(METRICS_DIR)
    time_list = metrics["Running Time"]

    start_time = time_list[0]
    end_time = time_list[-1]
    time_step = 0.01

    current_time = start_time
    time_chunks = [10.0]
    while current_time <= end_time:
        if current_time in time_list:
            current_time = round(current_time + time_step, 2)
        else:
            prev_time = round(current_time - time_step, 2)
            time_chunks.append(prev_time)
            current_time = round(current_time + 9*time_step, 2)
            time_chunks.append(current_time)
            
    # FIX 1: Sort the list and remove duplicates so the plot doesn't draw backwards
    return sorted(list(set(time_chunks)))

def get_time_series(start_time: float, end_time: float, time_step: float):
    time_list = []
    current_time = start_time
    while current_time <= end_time:
        time_list.append(current_time)
        current_time = round(current_time + time_step, 2)
    return time_list

def read_variables(time: float, variable_name: str):
    standard_path = f"{GROUND_TRUTH_DIR}/{variable_name}_{time}.npy"
    predicted_path = f"{PREDICTION_DIR}/{variable_name}_{time}_predicted.npy"

    if os.path.exists(predicted_path):
        if variable_name == "T":
            return np.load(predicted_path).reshape(-1)
        return np.load(predicted_path)
    elif os.path.exists(standard_path):
        return np.load(standard_path)
    else:
        raise FileNotFoundError(f"Neither {predicted_path} nor {standard_path} exists.")

# FIX 2: Helper to get the Ground Truth baseline to properly scale divergences up
def get_gt_baselines(time: float):
    prev_time = round(time - 0.01, 2)
    
    U = np.load(f"{GROUND_TRUTH_DIR}/U_{time}.npy")
    T = np.load(f"{GROUND_TRUTH_DIR}/T_{time}.npy").reshape(200, 200)
    U_prev = np.load(f"{GROUND_TRUTH_DIR}/U_{prev_time}.npy")
    T_prev = np.load(f"{GROUND_TRUTH_DIR}/T_{prev_time}.npy").reshape(200, 200)
    
    ux = U[..., 0].reshape(200, 200)
    uy = U[..., 1].reshape(200, 200)
    ux_prev = U_prev[..., 0].reshape(200, 200)
    
    rm = residual_mass(np.stack([ux, uy], axis=-1))
    rmom = residual_momentum(ux, ux_prev, uy, T)
    rh = residual_heat(ux, uy, T, T_prev)
    return rm, rmom, rh

def calculate_residuals(time_list, chunked=False):
    ux_matrix_prev = None
    t_matrix_prev = None

    res_mass = []
    res_mom = []
    res_heat = []
    for time in time_list:
        U_matrix: np.ndarray = read_variables(time, "U")
        t_matrix: np.ndarray = read_variables(time, "T").reshape(200,200)
        ux_matrix = U_matrix[..., 0].reshape(200,200)
        uy_matrix = U_matrix[..., 1].reshape(200,200)
        U_stacked = np.stack([ux_matrix, uy_matrix], axis=-1)

        if ux_matrix_prev is None and t_matrix_prev is None:
            ux_matrix_prev = ux_matrix
            t_matrix_prev = t_matrix
            continue
        elif chunked:
            prev_time = round(time - 0.01, 2)
            U_matrix_prev = read_variables(prev_time, "U")
            t_matrix_prev = read_variables(prev_time, "T").reshape(200,200)
            ux_matrix_prev = U_matrix_prev[..., 0].reshape(200,200)

        res_mass.append(residual_mass(U_stacked))
        res_mom.append(residual_momentum(ux_matrix, ux_matrix_prev, uy_matrix, t_matrix))
        res_heat.append(residual_heat(ux_matrix, uy_matrix, t_matrix, t_matrix_prev))

        if not chunked:
            ux_matrix_prev = ux_matrix
            t_matrix_prev = t_matrix
    return res_mass, res_mom, res_heat

def main():
    time_list = chunk_time_series()
    res_mass, res_mom, res_heat = calculate_residuals(time_list, chunked=True)
    
    # Grab the ground truth baselines for scaling
    first_time_evaluated = time_list[1]
    gt_mass_0, gt_mom_0, gt_heat_0 = get_gt_baselines(first_time_evaluated)
    
    # Scale against Ground Truth to ensure divergence visually spikes upwards
    t = np.array(time_list[1:])
    mass = np.array(res_mass) / gt_mass_0
    mom = np.array(res_mom) / gt_mom_0
    heat = np.array(res_heat) / gt_heat_0

    # Calculate moving averages
    window_size = 50  # Adjust this value to increase/decrease smoothing
    if len(t) > window_size:
        t_smooth = t[window_size - 1:]
        mass_smooth = moving_average(mass, n=window_size)
        mom_smooth = moving_average(mom, n=window_size)
        heat_smooth = moving_average(heat, n=window_size)
    else:
        t_smooth, mass_smooth, mom_smooth, heat_smooth = t, mass, mom, heat

    # --- Plotting (Nature style) ---
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.linewidth": 1.2,
        "axes.labelsize": 15,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    # Plot raw data as faint background lines
    ax.plot(t, mass, color=COL_BLUE_DARK, alpha=0.25, lw=1.0)
    ax.plot(t, mom, color=COL_ORANGE_DARK, alpha=0.25, lw=1.0)
    ax.plot(t, heat, color=COL_GREEN_DARK, alpha=0.25, lw=1.0)

    # Plot smoothed lines
    ax.plot(t_smooth, mass_smooth, color=COL_BLUE_DARK, lw=2.2, label="Mass Residual")
    ax.plot(t_smooth, mom_smooth, color=COL_ORANGE_DARK, lw=2.2, label="Momentum Residual")
    ax.plot(t_smooth, heat_smooth, color=COL_GREEN_DARK, lw=2.2, label="Heat Residual")

    # Axes styling
    ax.set_ylim(1e-1, 1e5)
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scaled Residual")
    ax.set_title("Residuals over Time")

    ax.grid(True, which="major", ls="--", lw=0.6, alpha=0.4)
    ax.grid(True, which="minor", ls=":", lw=0.5, alpha=0.25)

    # Clean legend
    leg = ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(0.8)

    # Tight layout & save
    fig.tight_layout()
    fig.savefig("residuals_over_time.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()