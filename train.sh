#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

config_files=(
    "configs/gazefollow.py"
    "configs/gazefollow_518.py"
    "configs/videoattentiontarget.py"
)

run_experiment() {
    local config="$1"
    echo "Running experiment with config: $config"
    python -u tools/train.py --config-file "$config" --num-gpu 2
}

for config in "${config_files[@]}"
do
    run_experiment "$config" &
    pid=$!
    wait "$pid"
    sleep 10
done
