from pathlib import Path
import warnings
import imageio
from typing import Dict, Union, List
import json
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from repitframework.config import BaseConfig, TrainingConfig

# Choose a clear scientific style
plt.style.use('seaborn-v0_8-darkgrid')

warning_string = (
    "\nData dimension mismatch:\n"
    "\tVariable shape: {}\n"
    "\tExpected data dimension: {}\n"
    "Please update data_dim in config if you want to visualize all dimensions.\n"
)

AXIS_LIST = ["x", "y", "z"]


def flip_and_reshape(data: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return data.reshape(ny, nx)


def process_variable(
    data_dict: Dict[str, np.ndarray],
    var: str,
    data_dim: int,
    nx: int,
    ny: int
) -> Dict[str, np.ndarray]:
    shape = data_dict[var].shape
    if len(shape) == 2:
        dim = shape[-1]
        if dim != data_dim:
            warnings.warn(warning_string.format(shape, data_dim))
        if dim == 1:
            data_dict[var] = flip_and_reshape(data_dict[var], nx, ny)
        else:
            for i in range(min(data_dim, len(AXIS_LIST))):
                key = f"{var}_{AXIS_LIST[i]}"
                data_dict[key] = flip_and_reshape(data_dict[var][:, i], nx, ny)
            del data_dict[var]
    elif len(shape) == 1:
        data_dict[var] = flip_and_reshape(data_dict[var], nx, ny)
    else:
        raise ValueError(f"Unexpected shape {shape} for '{var}'")
    return data_dict


def load_metrics(metrics_path: Path) -> Dict[str, List[float]]:
    data = defaultdict(list)
    if metrics_path.suffix == ".ndjson":
        with open(metrics_path) as f:
            for line in f:
                rec = json.loads(line)
                data[rec['key']].append(rec['value'])
        return data
    return json.loads(metrics_path.read_text())


def visualize_output(
    base_config: BaseConfig,
    timestamp: Union[int, float],
    np_data_dir: Path = None,
    data_vars: List[str] = None,
    save_name: str = "output",
    mode: str = "image",
    is_ground_truth: bool = True
) -> Union[bool, np.ndarray]:
    np_data_dir = Path(np_data_dir) if np_data_dir else base_config.assets_path
    data_vars = data_vars or base_config.extend_variables()
    out_dir = base_config.root_dir / "plots" / np_data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dim, ny, nx = base_config.data_dim, base_config.grid_y, base_config.grid_x
    fields = {}
    for var in data_vars:
        fname = f"{var}_{timestamp}{'_predicted'*(not is_ground_truth)}.npy"
        arr = np.load(np_data_dir / fname)
        fields[var] = arr
        fields = process_variable(fields, var, data_dim, nx, ny)

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, mat) in zip(axes, fields.items()):
        im = ax.imshow(mat, origin='lower', cmap='inferno', aspect='equal')
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(name, fontsize=12)
        ax.axis('off')

    fig.suptitle(f"t = {timestamp}s", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.92])

    if mode == 'image':
        fig.savefig(out_dir / f"{save_name}_{timestamp}.png", dpi=200)
        plt.close(fig)
        return True
    elif mode == 'rgb_array':
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)
        return rgb
    else:
        plt.close(fig)
        raise ValueError("mode must be 'image' or 'rgb_array'")


def get_probes_data(
    pred_time_list: List[float],
    full_time_list: List[float],
    ground_truth_dir: Path,
    prediction_dir: Path,
    plot_prediction_only: bool = False
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    probes = {var: {'ground_truth': defaultdict(list), 'predicted': defaultdict(list)}
              for var in ['T', 'U_x', 'U_y']}
    probes_labels = {'t1':39699, 't2':39499, 't3':39299, 'b1':299, 'b2':499, 'b3':699}

    timestamps = pred_time_list if plot_prediction_only else np.round(
        np.arange(min(full_time_list), max(full_time_list), round(full_time_list[1]-full_time_list[0], 2)), 2)

    for t in timestamps:
        gt_T = np.load(ground_truth_dir / f"T_{t}.npy"); gt_U = np.load(ground_truth_dir / f"U_{t}.npy")
        if t in pred_time_list:
            pred_T = np.load(prediction_dir / f"T_{t}_predicted.npy"); pred_U = np.load(prediction_dir / f"U_{t}_predicted.npy")
        else:
            pred_T = np.load(prediction_dir / f"T_{t}.npy"); pred_U = np.load(prediction_dir / f"U_{t}.npy")

        for label, idx in probes_labels.items():
            probes['T']['ground_truth'][label].append(float(gt_T[idx]))
            probes['T']['predicted'][label].append(float(pred_T[idx]))
            probes['U_x']['ground_truth'][label].append(float(gt_U[idx,0]))
            probes['U_x']['predicted'][label].append(float(pred_U[idx,0]))
            probes['U_y']['ground_truth'][label].append(float(gt_U[idx,1]))
            probes['U_y']['predicted'][label].append(float(pred_U[idx,1]))
    return probes


def quantitative_analysis(
    pred_time_list: List[float],
    full_time_list: List[float],
    ground_truth_dir: Path,
    prediction_dir: Path,
    save_name: str = "velocity-x",
    plot_prediction_only: bool = False
) -> None:
    probes = get_probes_data(pred_time_list, full_time_list, ground_truth_dir, prediction_dir, plot_prediction_only)
    is_temp = save_name.lower() in ['temperature', 't']
    var_key = 'T' if is_temp else save_name.replace('-', '_')

    fig, axs = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    for i, wall in enumerate(['ground_truth', 'predicted']):
        for label in ['t1','t2','t3'] if wall=='ground_truth' else ['t1','t2','t3']:
            axs[0].plot(probes[var_key][wall][label],
                        linestyle='-' if wall=='ground_truth' else '--',
                        label=f"{wall[0].upper()}{label}",
                        linewidth=2)
    axs[0].set_title('Top Wall'); axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(fontsize=8); axs[0].set_ylabel(save_name)
    if is_temp: axs[0].set_ylim(292,304)
    else: axs[0].set_ylim(-0.1,0.2)

    for i, wall in enumerate(['ground_truth', 'predicted']):
        for label in ['b1','b2','b3']:
            axs[1].plot(probes[var_key][wall][label],
                        linestyle='-' if wall=='ground_truth' else '--',
                        label=f"{wall[0].upper()}{label}",
                        linewidth=2)
    axs[1].set_title('Bottom Wall'); axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(fontsize=8); axs[1].set_ylabel(save_name)
    if is_temp: axs[1].set_ylim(290.5,293.5)
    else: axs[1].set_ylim(-0.1,0.04)
    axs[1].set_xlabel('Timesteps')

    out_dir = Path(str(prediction_dir).replace('Assets','plots'))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(out_dir / f"{save_name}_analysis.png", dpi=200)
    plt.close(fig)


def plot_MAE(
    pred_time_list: List[float],
    ground_truth_dir: Path,
    prediction_dir: Path,
    var_name: str = "velocity-x"
) -> None:
    prefix = {'velocity-x':'U', 'velocity-y':'U', 'temperature':'T'}[var_name]
    idx = {'velocity-x':0, 'velocity-y':1}.get(var_name, None)

    errors, times = [], []
    for t in pred_time_list:
        gt = np.load(ground_truth_dir / f"{prefix}_{t}.npy"); pr = np.load(prediction_dir / f"{prefix}_{t}_predicted.npy")
        if idx is not None:
            gt = gt[:,idx]; pr = pr[:,idx]
        else:
            pr = pr.flatten()
        ae = np.abs(gt-pr)
        errors.append(ae.max()); times.append(t)
    max_err = max(errors); max_t = times[errors.index(max_err)]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(times, errors, linewidth=2.5, label='MAE')
    ax.scatter([max_t],[max_err], color='red', s=80, label=f'Max AE={max_err:.3f}@{max_t:.2f}s')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Max AE')
    ax.set_title(f'MAE for {var_name}'); ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10)

    out_dir = Path(str(prediction_dir).replace('Assets','plots'))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{var_name}_MAE.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_residual_change(
    running_times: List[float],
    relative_residual: List[float],
    residual_limit: float=5.0,
    save_name: str="relative_residual",
    save_path: Union[str, Path]=None
) -> None:
    save_dir = Path(save_path) if save_path else Path('.')
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(running_times, [1]*len(running_times), ':k', label='Reference')
    ax.plot(running_times, relative_residual, linewidth=2.5, label='Residual')
    ax.set_yscale('log'); ax.set_ylim(0.1,100)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Scaled Residual')
    ax.set_title(f'Relative Residual (limit={residual_limit})')
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{save_name}.png", dpi=200)
    plt.close(fig)


def still_comparisons(
    prediction_dir: Union[str,Path],
    ground_truth_dir: Union[str,Path],
    time_list: List[float]=[20,60,110],
    temp_profiles: List[float]=[288.15,307.75]
) -> None:
    pred_dir, gt_dir = Path(prediction_dir), Path(ground_truth_dir)
    cmap_temp, cmap_vel = 'turbo','viridis'
    n = len(time_list)
    fig, axs = plt.subplots(n,4,figsize=(16,4*n),constrained_layout=True)
    for i,t in enumerate(time_list):
        gt_t = np.load(gt_dir/f"T_{t}.npy").reshape(200,200)
        gt_u = np.load(gt_dir/f"U_{t}.npy")[:,0].reshape(200,200)
        try:
            pr_t = np.load(pred_dir/f"T_{t}_predicted.npy").reshape(200,200)
            pr_u = np.load(pred_dir/f"U_{t}_predicted.npy")[:,0].reshape(200,200)
        except:
            pr_t, pr_u = gt_t, gt_u
        for j,(data,cmap,title) in enumerate(
            [(gt_t,cmap_temp,'GT-T'),(gt_u,cmap_vel,'GT-U'),
             (pr_t,cmap_temp,'PR-T'),(pr_u,cmap_vel,'PR-U')]):
            ax=axs[i,j]; im=ax.imshow(data,origin='lower',cmap=cmap)
            if j%2==0: fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            ax.set_title(f"{title}@{t}s"); ax.axis('off')
    out=Path(str(prediction_dir).replace('Assets','plots'))
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/"still_comparisons.png",dpi=200); plt.close(fig)


def save_loss(
    training_config: TrainingConfig,
    save_initial_losses: bool=False,
    merge_initial_losses: bool=False
) -> None:
    metrics = load_metrics(training_config.model_dir/"training_metrics.ndjson")
    train,val=metrics['Training Loss'],metrics['Validation Loss']
    if save_initial_losses:
        idx=val.index(min(val))+1
        init={'Training Loss':train[:idx],'Validation Loss':val[:idx]}
        (training_config.model_dir/'initial_losses.json').write_text(json.dumps(init,indent=4))
    if merge_initial_losses:
        init=json.loads((training_config.model_dir/'initial_losses.json').read_text())
        train,val=init['Training Loss']+train,init['Validation Loss']+val
    epochs=np.arange(1,len(train)+1)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(epochs[1:],train[1:],linewidth=2.5,label='Train')
    ax.plot(epochs[1:],val[1:],linewidth=2.5,label='Val')
    ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss');ax.grid(True,linestyle='--',alpha=0.6);ax.legend()
    out=Path(str(training_config.model_dir).replace('ModelDump','plots'))/'loss'
    out.mkdir(parents=True,exist_ok=True)
    tstamp=datetime.now().strftime('%b-%d_%H%M')
    fig.savefig(out/f"loss_{tstamp}.png",dpi=200); plt.close(fig)


def plot_everything(plot_start_time: float=10.0, plot_end_time: float=110.0) -> None:
    base_cfg=BaseConfig(); train_cfg=TrainingConfig()
    solver=base_cfg.solver_dir
    model_dump=Path(str(solver).replace('Solvers','ModelDump'))
    gt_dir=Path(str(solver).replace('Solvers','Assets')+'_backup')
    pr_dir=Path(str(solver).replace('Solvers','Assets'))
    plots_dir=Path(str(solver).replace('Solvers','plots'))
    metrics=load_metrics(model_dump/'prediction_metrics.ndjson')
    times=metrics['Running Time']; full=np.round(np.arange(plot_start_time,plot_end_time,0.01),2).tolist()
    plot_MAE(times,gt_dir,pr_dir,'velocity-x')
    plot_MAE(times,gt_dir,pr_dir,'temperature')
    plot_residual_change(times,metrics['Relative Residual Mass'],5,save_path=plots_dir)
    still_comparisons(pr_dir,gt_dir,[ (plot_start_time+plot_end_time)/2,plot_end_time ],[train_cfg.left_wall_temperature,train_cfg.right_wall_temperature])

def plot_spectral_analysis(
    Ux_true: np.ndarray,
    Ux_pseudo: np.ndarray,
    Uy_true: np.ndarray,
    Uy_pseudo: np.ndarray,
    T_true: np.ndarray,
    T_pseudo: np.ndarray
) -> None:
    """
    Compute and plot radial spectra for velocity and temperature comparisons.
    """
    # Compute 2D FFTs
    fft = lambda x: np.fft.fft2(x)
    FX_true, FX_pseudo = fft(Ux_true), fft(Ux_pseudo)
    FY_true, FY_pseudo = fft(Uy_true), fft(Uy_pseudo)
    FT_true, FT_pseudo = fft(T_true), fft(T_pseudo)

    # Energy fields
    E_vel_true = np.abs(FX_true)**2 + np.abs(FY_true)**2
    E_vel_pseudo = np.abs(FX_pseudo)**2 + np.abs(FY_pseudo)**2
    E_T_true = np.abs(FT_true)**2
    E_T_pseudo = np.abs(FT_pseudo)**2

    # Radial spectrum
    def radial(E: np.ndarray) -> np.ndarray:
        # Shift: 
        E = np.fft.fftshift(E)
        nx, ny = E.shape
        cx, cy = nx // 2, ny // 2
        y, x = np.indices(E.shape)
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
        tbin = np.bincount(r.ravel(), E.ravel())
        nr = np.bincount(r.ravel())
        return tbin / (nr + 1e-10)

    E_vel_true_r = radial(E_vel_true)
    E_vel_pseudo_r = radial(E_vel_pseudo)
    E_T_true_r = radial(E_T_true)
    E_T_pseudo_r = radial(E_T_pseudo)
    k = np.arange(len(E_T_true_r))

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 5), constrained_layout=True)
    cmap = plt.get_cmap('tab10')

    # Difference
    axs[0, 0].plot(k, E_vel_pseudo_r - E_vel_true_r, label='Velocity Δ', linewidth=2, color=cmap(0))
    axs[0, 1].plot(k, E_T_pseudo_r - E_T_true_r, label='Temp Δ', linewidth=2, color=cmap(1))

    # Ratio
    axs[1, 0].plot(k, E_vel_pseudo_r / (E_vel_true_r + 1e-12), label='Velocity Ratio', linewidth=2, color=cmap(2))
    axs[1, 1].plot(k, E_T_pseudo_r / (E_T_true_r + 1e-12), label='Temp Ratio', linewidth=2, color=cmap(3))

    for ax in axs.flatten():
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber k', fontsize=10)
        ax.set_ylabel('Energy', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    axs[0, 0].set_title('Velocity Spectrum Difference', fontsize=12, fontweight='bold')
    axs[0, 1].set_title('Temperature Spectrum Difference', fontsize=12, fontweight='bold')
    axs[1, 0].set_title('Velocity Spectrum Ratio', fontsize=12, fontweight='bold')
    axs[1, 1].set_title('Temperature Spectrum Ratio', fontsize=12, fontweight='bold')

    # plt.savefig('spectral_analysis.png', dpi=200)
    plt.show()
    
if __name__=='__main__':
    plot_everything()

