#!/bin/bash

mkdir -p datasets/TartanAir

DATA_ROOT=/media/vmo/KINGSTON/SLAM_Datasets/tartan_air

for scene in "$DATA_ROOT"/*; do
    name=$(basename "$scene")
    mkdir -p "datasets/TartanAir/$name"
    ln -sfn "$scene" "datasets/TartanAir/$name/$name"
done
