import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation


def load_gt_traj(gt_file, est_traj):
    """
    GT format:
        x y z qw qx qy qz

    Uses timestamps from estimated trajectory.
    Assumes GT and estimate have identical row count.
    """
    data = np.loadtxt(gt_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) != est_traj.num_poses:
        raise ValueError(
            f"GT length ({len(data)}) != EST length ({est_traj.num_poses})"
        )
    xyz = data[:, 0:3]
    quat_wxyz = data[:, 3:7]
    return PoseTrajectory3D(
        positions_xyz=xyz,
        orientations_quat_wxyz=quat_wxyz,
        timestamps=est_traj.timestamps.copy(),
    )


def compute_ate_rmse(gt_traj, est_traj):
    result = main_ape.ape(
        gt_traj,
        est_traj,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )
    return result.stats["rmse"]


def est_to_gt_name(est_name):
    """
    Example:
        TartanAir_Abandonedfactory_Easy_P011_Trial01.txt
    ->
        TartanAir_Abandonedfactory_Easy_P011_GT.txt
    """
    parts = est_name.split("_")
    return "_".join(parts[:-1]) + "_GT.txt"


def collect_ates(est_dir, gt_dir, tag=""):
    ates = []
    est_files = sorted(f for f in os.listdir(est_dir) if f.endswith(".txt"))
    for fname in tqdm(est_files, desc=tag, ncols=100):
        est_path = os.path.join(est_dir, fname)
        gt_path = os.path.join(gt_dir, est_to_gt_name(fname))
        if not os.path.exists(gt_path):
            continue
        try:
            est_traj = file_interface.read_tum_trajectory_file(est_path)
            gt_traj = load_gt_traj(gt_path, est_traj)
            ate = compute_ate_rmse(gt_traj, est_traj)
            ates.append(ate)
        except Exception as e:
            tqdm.write(f"[skip] {fname}: {e}")
    return np.array(ates)


def cdf_curve(values):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def compute_auc(ates, max_x=1.0):
    """
    AUC of the empirical CDF from 0 to max_x.
    Normalized to [0, 1].
    """
    if len(ates) == 0:
        return np.nan
    x, y = cdf_curve(ates)

    #
    mask = x <= max_x
    x = x[mask]
    y = y[mask]

    #
    x = np.concatenate(([0.0], x))
    y = np.concatenate(([0.0], y))
    if x[-1] < max_x:
        x = np.concatenate((x, [max_x]))
        y = np.concatenate((y, [y[-1]]))
    auc = np.trapezoid(y, x) / max_x
    return auc


def plot_cdf(curves, saved_path=None, xlim=(0, 1.0)):
    """
    curves:
        dict[str, np.ndarray]

    Example:
        {
            "DROID": droid_ates,
            "Ours": ours_ates,
            "Exp3": exp3_ates,
        }
    """
    max_x = xlim[1]
    plt.figure(figsize=(6, 4))
    for label, ates in curves.items():
        if len(ates) == 0:
            continue
        x, y = cdf_curve(ates)
        plt.step(x, y, where="post", label=label)
        auc = compute_auc(ates, max_x=max_x)
        print(f"{label}: AUC={auc:.4f}")
    plt.xlabel("ATE [m]")
    plt.ylabel("% runs")
    plt.xlim(*xlim)
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if saved_path is not None:
        plt.savefig(saved_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    #
    base_dir = "/home/hapq/Desktop/dpvo_stuff"
    gt_dir = os.path.join(base_dir, "gt_trajs_fixed")

    #
    experiments = {
        "Pretrain": os.path.join(
            base_dir,
            "default_weight/eval_res/dpvo_ckpt/saved_trajectories",
        ),
        "demo_train2_130000": os.path.join(
            base_dir,
            "demo_train/eval_res/05-10-12PM_demo_train2_130000_ckpt/saved_trajectories",
        ),
        "demo_train2_120000": os.path.join(
            base_dir,
            "demo_train/eval_res/05-10-11PM_demo_train2_120000_ckpt/saved_trajectories",
        ),
        "demo_train2_080000": os.path.join(
            base_dir,
            "demo_train/eval_res/05-10-07PM_demo_train2_080000_ckpt/saved_trajectories",
        ),
    }
    curves = {}
    for name, est_dir in experiments.items():
        curves[name] = collect_ates(est_dir, gt_dir, tag=name)
    plot_cdf(
        curves,
        saved_path=os.path.join(base_dir, "ate_cdf.png"),
    )
