from pathlib import Path
import numpy as np
import argparse

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default=str(Path.cwd() / "TartanAirResults"),
        help="Path to TartanAirResults directory",
    )
    args = parser.parse_args()

    #
    results_dir = Path(args.results_dir).expanduser()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    #
    scores = []
    for f in sorted(results_dir.glob("*.txt")):
        with open(f, "r") as fp:
            values = np.array(
                [float(x) for x in fp.read().strip().split(",") if x.strip()],
                dtype=np.float64,
            )
        scores.append((f.name, values.mean(), np.median(values)))
    if not scores:
        raise RuntimeError(f"No result files found in {results_dir}")

    #
    print(f"Using results directory: {results_dir}")
    print()
    for checkpoint, mean_rmse, median_rmse in scores:
        print(f"{checkpoint}: mean={mean_rmse:.6f}, median={median_rmse:.6f}")
    best_median_checkpoint, best_mean, best_median = min(scores, key=lambda x: x[2])
    best_mean_checkpoint, best_mean_rmse, best_mean_median = min(scores, key=lambda x: x[1])
    print()
    print(f"Best checkpoint (median): {best_median_checkpoint}")
    print(f"  mean RMSE   = {best_mean:.6f}")
    print(f"  median RMSE = {best_median:.6f}")
    print(f"Best checkpoint (mean): {best_mean_checkpoint}")
    print(f"  mean RMSE   = {best_mean_rmse:.6f}")
    print(f"  median RMSE = {best_mean_median:.6f}")
