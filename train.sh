#!/bin/bash

config_files=(
    "configs\gazefollow.py"
    "configs\gazefollow_518.py"
)

log_file="train.log"
> "$log_file"

run_experiment() {
    local config="$1"
    echo "Running experiment with config: $config" | tee >> "$log_file"
    python -u tools/train.py --config-file "$config" --num-gpu 2 2>&1 | tee >> "$log_file"
}

for config in "${config_files[@]}"
do
    run_experiment "$config" &
    pid=$!
    wait "$pid"
    sleep 10
done
