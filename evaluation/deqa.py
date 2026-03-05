import requests
import torch
from transformers import AutoModelForCausalLM
import torch
import argparse
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
import os
import natsort

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_prefix', type=str, required=True, help='Path to images directory or prefix')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model directory or prefix')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    img_prefix = args.img_prefix
    step = img_prefix.split('/')[-1].split('-')[0]
    print("Image path:", img_prefix)
    
    med_config = os.path.join(os.path.dirname(args.model_path), 'med_config.json')
    img_list = natsorted([os.path.join(img_prefix, img) for img in os.listdir(img_prefix) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    # The inputs should be a list of multiple PIL images
    score_sum = 0.0
    batch_size = 20
    num_imgs = len(img_list)

    for i in tqdm(range(0, num_imgs, batch_size)):
        actual_bs = min(batch_size, num_imgs - i)
        batch_paths = img_list[i:i+actual_bs]   # 注意这里批次实际长度可能小于8
        batch_imgs = [Image.open(p) for p in batch_paths]
        with torch.no_grad():
            batch_scores = model.score(batch_imgs)
            batch_scores = batch_scores.tolist()
            # print(f"Image: {img_path}, Score: {score}")
            for score in batch_scores:
                score_sum += score
    # print("Total score:", score_sum)
    print(f"**{step}** Average score:", score_sum / len(img_list))