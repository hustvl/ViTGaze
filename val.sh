#!/bin/bash
# [Usage] bash val.sh [$config_file] [$weight_file] [$(gf/vat)]; e.g.
# bash val.sh configs/gazefollow_518.py output/gazefollow_518/model_final.pth gf
# bash val.sh configs/videoattentiontarget.py output/videoattentiontarget/model_final.pth vat

config_file="$1"
checkpoint="$2"

if [ "$3" = "gf" ]; then
    evaluater="tools/eval_on_gazefollow.py"
elif [ "$3" = "vat" ]; then
    evaluater="tools/eval_on_video_attention_target.py"
else
    echo "Invalid dataset"
    exit 1
fi

echo "Evaluating with:"
echo "config: $config_file"
echo "checkpoint: $checkpoint"
python -u $evaluater --config_file $config_file --model_weights "$checkpoint" --use_dark_inference
