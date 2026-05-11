#!/bin/bash

# Usage:
# ./collect_tartan_gt.sh /path/to/dataset /path/to/output

set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_path> <output_path>"
    exit 1
fi

IN_DIR="$1"
OUT_DIR="$2"
mkdir -p "$OUT_DIR"
find "$IN_DIR" -type f -path "*/P*/pose_left.txt" | while read -r f; do
    rel="${f#$IN_DIR/}"
    env=$(echo "$rel" | cut -d/ -f1)
    diff=$(echo "$rel" | cut -d/ -f2)
    seq=$(echo "$rel" | cut -d/ -f3)
    env_cap=$(echo "$env" | awk -F'_' '{
        for (i = 1; i <= NF; i++) {
            $i = toupper(substr($i,1,1)) substr($i,2)
        }
        for (i = 1; i <= NF; i++) {
            printf "%s%s", $i, (i < NF ? "_" : "")
        }
        printf "\n"
    }')

    out_file="${OUT_DIR}/TartanAir_${env_cap}_${diff}_${seq}_GT.txt"
    cp "$f" "$out_file"
    echo "Copied -> $out_file"
done
