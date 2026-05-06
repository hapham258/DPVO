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
from dpvo.utils import Timer

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):
    #
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    #
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    #
    slam = None
    for t_ns, imfile in enumerate(tqdm(image_list, desc="DPVO", unit="frame")):
        #
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])
        intrinsics = np.array([fx, fy, cx, cy])    
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        #
        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)
        slam(t_ns, image, intrinsics)

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
