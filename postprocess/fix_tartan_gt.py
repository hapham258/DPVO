from pathlib import Path
import argparse
import numpy as np

PERM = [1, 2, 0, 4, 5, 3, 6]


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Apply Tartan NED->XYZ column permutation to all *_GT.txt files."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing *_GT.txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to save converted files")
    args = parser.parse_args()

    #
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #
    files = sorted(input_dir.glob("*_GT.txt"))
    for f in files:
        traj = np.loadtxt(f)
        traj_fixed = traj[:, PERM]
        out_file = output_dir / f.name
        np.savetxt(out_file, traj_fixed, fmt="%.10f")
        print(f"Saved {out_file}")
    print(f"Done: {len(files)} files converted")
