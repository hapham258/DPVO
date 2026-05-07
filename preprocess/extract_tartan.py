import os
import argparse
import zipfile
import shutil


def unzip_and_flatten(root_dir):
    temp_dir = os.path.join(root_dir, "_tmp_unzip")
    os.makedirs(temp_dir, exist_ok=True)

    #
    for f in os.listdir(root_dir):
        if not f.endswith(".zip"):
            continue
        zip_path = os.path.join(root_dir, f)
        print(f"Extracting: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)
            os.remove(zip_path)
            print(f"Removed: {zip_path}")
        except Exception as e:
            print(f"Failed extracting {zip_path}: {e}")

    #
    for item in os.listdir(temp_dir):
        src = os.path.join(temp_dir, item)
        dst = os.path.join(root_dir, item)
        if os.path.exists(dst):
            print(f"Skipping existing: {dst}")
            continue
        print(f"Moving: {src} -> {dst}")
        shutil.move(src, dst)

    # #
    # shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output dataset folder")
    args = parser.parse_args()
    unzip_and_flatten(args.out)
