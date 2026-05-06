import os
from pathlib import Path
from itertools import chain
from tqdm import tqdm

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP, save_ply


def load_calib(calib_path):
    with open(calib_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    vals = lines[0].split()
    return {
        "model": vals[0],
        "fx": float(vals[1]),
        "fy": float(vals[2]),
        "cx": float(vals[3]),
        "cy": float(vals[4]),
        "dist": np.array([float(x) for x in vals[5:]], dtype=np.float32),
        "orig_w": int(lines[1].split()[0]),
        "orig_h": int(lines[1].split()[1]),
        "crop_mode": lines[2],
        "out_w": int(lines[3].split()[0]),
        "out_h": int(lines[3].split()[1]),
    }


@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):
    #
    cal = load_calib(calib)
    W, H = cal["out_w"], cal["out_h"]

    #
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    #
    slam = None
    K = np.array([
        [cal["fx"], 0, cal["cx"]],
        [0, cal["fy"], cal["cy"]],
        [0, 0, 1],
    ], dtype=np.float32)
    D = cal["dist"].reshape(-1, 1)
    map1, map2 = None, None
    pbar = tqdm(image_list, desc="DPVO", unit="frame", leave=False)
    for t_ns, imfile in enumerate(pbar):
        #
        image = cv2.imread(str(imfile))
        orig_w, orig_h = cal["orig_w"], cal["orig_h"]
        out_w, out_h = cal["out_w"], cal["out_h"]
        if cal["model"] == "Pinhole":
            fx, fy = cal["fx"], cal["fy"]
            cx, cy = cal["cx"], cal["cy"]
            if out_w != orig_w or out_h != orig_h:
                x = (orig_w - out_w) // 2
                y = (orig_h - out_h) // 2
                image = image[y:y + out_h, x:x + out_w]
                cx -= x
                cy -= y
        elif cal["model"] == "RadTan":
            if map1 is None or map2 is None:
                new_K, _ = cv2.getOptimalNewCameraMatrix(
                    K,
                    D,
                    (orig_w, orig_h),
                    0,
                    (out_w, out_h),
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    K,
                    D,
                    None,
                    new_K,
                    (out_w, out_h),
                    cv2.CV_16SC2,
                )
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            fx = new_K[0, 0]
            fy = new_K[1, 1]
            cx = new_K[0, 2]
            cy = new_K[1, 2]
        elif cal["model"] == "EquiDistant":
            if map1 is None or map2 is None:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K,
                    D,
                    (orig_w, orig_h),
                    np.eye(3, dtype=np.float32),
                    None,
                    0.0,
                    (out_w, out_h),
                )
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K,
                    D,
                    np.eye(3, dtype=np.float32),
                    new_K,
                    (out_w, out_h),
                    cv2.CV_16SC2,
                )
            image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            fx = new_K[0, 0]
            fy = new_K[1, 1]
            cx = new_K[0, 2]
            cy = new_K[1, 2]
        else:
            raise ValueError(f"Unsupported camera model: {cal['model']}")
        intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device="cuda")

        #
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        image = torch.from_numpy(image).permute(2,0,1).cuda()

        #
        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)
        if not slam(t_ns, image, intrinsics):
            print("Exit main loop.")
            pbar.close()
            break

    #
    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    result = slam.terminate()
    print("DPVO finished.")
    return result, (points, colors, (*intrinsics, H, W))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_ply', action="store_true")
    parser.add_argument('--save_colmap', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    print("Running with config...")
    print(cfg)

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, args.imagedir, args.calib, args.stride, args.skip, args.viz, args.timeit)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    if args.save_ply:
        save_ply(args.name, points, colors)
    if args.save_colmap:
        save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)
    if args.save_trajectory:
        Path("saved_trajectories").mkdir(exist_ok=True)
        file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}.txt", trajectory)
    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"DPVO Trajectory Prediction for {args.name}", filename=f"trajectory_plots/{args.name}.pdf")
