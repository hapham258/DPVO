import os
from collections import defaultdict

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


def gt_to_seq_name(gt_name):
    """
    Examples:
        TartanAir_Abandonedfactory_Easy_P011_GT.txt -> Abandonedfactory_Easy
        TartanAir_ME000_GT.txt                      -> ME000
    """
    #
    stem = os.path.splitext(gt_name)[0]
    if not stem.startswith("TartanAir_") or not stem.endswith("_GT"):
        raise ValueError(f"Unexpected GT filename: {gt_name}")

    #
    core = stem[len("TartanAir_") : -len("_GT")]
    parts = core.split("_")
    if len(parts) == 1:
        return parts[0]
    if parts[-1].startswith("P"):
        return "_".join(parts[:-1])
    return "_".join(parts)


def collect_ates(est_dir, gt_dir, tag=""):
    """
    Single pass:
    - returns flat ATEs (for CDF)
    - returns per-sequence ATE lists
    """

    curves = []
    seq_to_ates = defaultdict(list)
    est_files = sorted(f for f in os.listdir(est_dir) if f.endswith(".txt"))
    for fname in tqdm(est_files, desc=tag, ncols=100):
        est_path = os.path.join(est_dir, fname)
        gt_name = est_to_gt_name(fname)
        gt_path = os.path.join(gt_dir, gt_name)
        if not os.path.exists(gt_path):
            continue
        try:
            est_traj = file_interface.read_tum_trajectory_file(est_path)
            gt_traj = load_gt_traj(gt_path, est_traj)
            ate = compute_ate_rmse(gt_traj, est_traj)
            curves.append(ate)
            seq_name = gt_to_seq_name(gt_name)
            seq_to_ates[seq_name].append(ate)
        except Exception as e:
            tqdm.write(f"[skip] {fname}: {e}")
    return np.array(curves), dict(seq_to_ates)


def save_seq_table(results_dict, pdf_path):
    """
    Vertical table:
    Rows = sequences
    Columns = methods
    Last row = median over sequences (per method)
    Best (lowest RMSE) highlighted per row AND last row.
    """

    #
    all_seqs = sorted(
        {seq for exp_dict in results_dict.values() for seq in exp_dict.keys()}
    )
    methods = sorted(results_dict.keys())
    columns = ["Sequence"] + methods
    rows = []
    method_vals_across_seqs = {m: [] for m in methods}
    numeric_rows = []
    for seq in all_seqs:
        row = [seq]
        numeric_row = []
        for method in methods:
            vals = results_dict[method].get(seq, [])
            med = np.median(vals) if len(vals) else np.nan
            numeric_row.append(med)
            if len(vals):
                method_vals_across_seqs[method].append(med)
            row.append(f"{med:.3f}")
        rows.append(row)
        numeric_rows.append(numeric_row)

    #
    median_row = ["Median"]
    median_values = []
    for method in methods:
        vals = method_vals_across_seqs[method]
        med = np.median(vals) if len(vals) else np.nan
        median_values.append(med)
        median_row.append(f"{med:.3f}")
    rows.append(median_row)
    numeric_rows.append(median_values)

    #
    fig_w = max(10, 0.8 * len(columns))
    fig_h = max(3, 0.5 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    #
    for i, numeric_row in enumerate(numeric_rows):
        valid = [(j, v) for j, v in enumerate(numeric_row) if not np.isnan(v)]
        if not valid:
            continue
        best_j = min(valid, key=lambda x: x[1])[0]
        cell = table[(i + 1, best_j + 1)]
        cell.set_text_props(weight="bold")
    last_row_idx = len(rows)
    for j in range(len(columns)):
        cell = table[(last_row_idx, j)]
        cell.set_text_props(fontstyle="italic")
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


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
    gt_dir = os.path.join(base_dir, "eval_gt_trajs_fixed")

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

    #
    curves = {}
    sequence_results = {}
    for name, est_dir in experiments.items():
        curves[name], sequence_results[name] = collect_ates(est_dir, gt_dir, tag=name)

    #
    save_seq_table(
        sequence_results,
        os.path.join(base_dir, "seq_median_table.pdf"),
    )
    plot_cdf(
        curves,
        saved_path=os.path.join(base_dir, "ate_cdf.png"),
    )
