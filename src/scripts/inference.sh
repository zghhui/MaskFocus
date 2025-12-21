#!/bin/bash
WORK_DIR="$(pwd)"
ROOT_DIR="$(dirname "$WORK_DIR")"
cd $ROOT_DIR/src/maskfocus/src
export PYTHONPATH="$ROOT_DIR/src/maskfocus/src:${PYTHONPATH}"

output_dir=$ROOT_DIR/output/inference
base_model_path=models--MeissonFlow--Meissonic
test_model_path=models--zghhui--Meissonic_MaskFocus_HPS
accelerate launch infer/inference.py \
    --model_path $base_model_path \
    --base_model_path $base_model_path \
    --output_dir $output_dir \
    --prompts_file $ROOT_DIR/src/maskfocus/src/infer/test_data.txt