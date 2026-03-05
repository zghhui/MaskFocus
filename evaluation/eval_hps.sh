
#!/bin/bash

####################################### HPS #######################################
# 1. inference

output_dir=result/hpsv2/step-100
base_model_path=model/models--MeissonFlow--Meissonic
test_model_path=checkpoint-dir

accelerate launch inference.py \
    --model_path $test_model_path \
    --base_model_path $base_model_path \
    --output_dir $output_dir \
    --prompts_file hpsv2/hps2_1_prompts.txt

# 2. eval hpsv2
cd hpsv2
MODEL_PATH="model/models--xswu--HPSv2/HPS_v2.1_compressed.pt"
MODEL_CONFIG="hpsv2/hps2_1_prompts.json"

# 1. 执行评估图像
echo "======= Starting HPSv2.1 image evaluation for STEP $STEP ======="
python evaluation.py --data-type benchmark \
    --data-path  "$MODEL_CONFIG" \
    --image-path "$output_dir" \
    --checkpoint "$MODEL_PATH"
echo "===============================================================" 