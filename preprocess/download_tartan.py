import os
import shutil
import argparse
import yaml
from huggingface_hub import snapshot_download


def build_file_list(cfg, type):
    env = cfg["env"]
    diffs = cfg["difficulty"]
    modalities = cfg["modality"]
    cams = cfg["camera_name"]
    files = []
    for diff in diffs:
        if type == "tartan_air":
            diff_dir = diff
        elif type == "tartan_air2":
            diff_dir = f"Data_{diff}"
        for cam in cams:
            for mod in modalities:
                files.append(f"{env}/{diff_dir}/{mod}_{cam}.zip")
    return files


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config")
    parser.add_argument("--out", type=str, required=True, help="Output dataset folder")
    args = parser.parse_args()

    #
    dataset_dir = args.out
    os.makedirs(dataset_dir, exist_ok=True)

    #
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    all_files = []
    dataset_type = cfg["dataset_type"]
    print(f"Dataset type: {dataset_type}")
    if dataset_type == "tartan_air":
        repo_id = "theairlabcmu/tartanair"
    elif dataset_type == "tartan_air2":
        repo_id = "theairlabcmu/tartanair2"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}.")
    for seq in cfg["sequence"]:
        all_files.extend(build_file_list(seq, dataset_type))

    #
    missing_files = []
    for file in all_files:
        flat_file = "_".join(file.split("/"))
        flat_path = os.path.join(dataset_dir, flat_file)
        if os.path.exists(flat_path):
            print(f"Skip (exists): {flat_path}")
        else:
            missing_files.append(file)

    #
    if missing_files:
        print("Downloading missing files:")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=missing_files,
            local_dir=dataset_dir,
            max_workers=8,
        )
    else:
        print("All files already exist.")

    #
    for file in missing_files:
        path = os.path.join(dataset_dir, file)
        parts = file.split(os.sep)
        flat_file = "_".join(parts[-3:])
        flat_path = os.path.join(dataset_dir, flat_file)
        shutil.move(path, flat_path)
