#!/bin/bash

####################################### Geneval #######################################
# 1. inference
steps=100
CFG=5
base_model_path="models--MeissonFlow--Meissonic"
prompts_file="geneval/generation_prompts.txt"

echo "======= Running inference for STEP $STEP ======="
output_dir="result/geneval/step-$steps/samples"
test_model_path="eval_ckpt_dir"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch inference.py  \
    --model_path "$test_model_path" \
    --CFG $CFG\
    --base_model_path "$base_model_path" \
    --output_dir "$output_dir" \
    --prompts_file "$prompts_file"

# 2. eval geneval
cd github/geneval
conda activate geneval

MODEL_CONFIG="geneval/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
OUTFILE="$output_dir/epoch_0_results.jsonl"
Geneval_MODEL_PATH="/home/panyaning/sdpdev-fs-pyn/huggingface/model/geneval"

echo "======= Starting image evaluation for STEP $STEP on GPU $GPU_IDX ======="
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_images.py \
    --imagedir "$IMAGE_DIR" \
    --model-config "$MODEL_CONFIG" \
    --outfile "$OUTFILE" \
    --model-path "$Geneval_MODEL_PATH"

python evaluation/summary_scores.py "$OUTFILE"
echo "======= Summary for STEP $STEP completed ======="