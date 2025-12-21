#!/bin/bash
ROOT_DIR="$(pwd)"
echo "$ROOT_DIR"
filename="7B_$(basename $(dirname $(pwd)))"

# conda activate meissonic_rl

cd $ROOT_DIR/maskfocus/src
RUN_NAME="mask-r1"

export DEBUG_MODE="true"
export LOG_PATH="./outputs/debug.txt"
# export NCCL_DEBUG=INFO

QWEN_PATH="/home/bingxing2/ailab/quyichang/zgh/data/pretrain_model/models--MeissonFlow--Meissonic"
# HF_DATASET="../../../data/flowgrpo_train_metadata.json" 
# HF_DATASET="../../../data/geneval_style_data.json" 
HF_DATASET="../../../data/hps_train_final.json" 

OUTPUT_DIR="../../../outputs/${RUN_NAME}" 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch --config_file "../configs/zero2.yaml" \
open_r1/grpo.py --use_vllm False \
--output_dir $OUTPUT_DIR \
--model_name_or_path $QWEN_PATH \
--dataset_name $HF_DATASET \
--max_prompt_length 512 \
--max_completion_length 1024 \
--temperature 1.0 \
--num_generations 2 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--logging_steps 1 \
--torch_dtype bfloat16 \
--report_to tensorboard \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_steps 1200 \
--run_name $RUN_NAME \
--save_steps 100 \
--new_generations_image 1 \
--gen_steps 64 \
--cirtical_steps 6 \
--resolution 1024 \
--cfg_weight 5 \
--reward_funcs hpsv2 \
--beta 0.01 \
--tf32 true \
--learning_rate 2e-6 \
--hps_ckpt_path /home/bingxing2/ailab/quyichang/zgh/data/pretrain_model/models--xswu--HPSv2/HPS_v2.1_compressed.pt \
--img_save_dir ../../../outputs/img_save \
--save_only_model true