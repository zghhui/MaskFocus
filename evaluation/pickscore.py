# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

def calc_probs(model, processor, prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        # probs = torch.softmax(scores, dim=-1)
    
    return scores.cpu().tolist()




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
    parser.add_argument('--prompt_file', type=str, required=True, help='Text prompt to evaluate against images')
    args = parser.parse_args()

    # load model
    device = "cuda"
    processor_name_or_path = "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "models--yuvalkirstain--PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    img_prefix = args.img_prefix
    prompt_path = args.prompt_file

    print("Image path:", img_prefix)
    
    img_list = natsorted([os.path.join(img_prefix, img) for img in os.listdir(img_prefix) if img.endswith(('.png', '.jpg', '.jpeg'))])
    with open(prompt_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    # The inputs should be a list of multiple PIL images
    score_sum = 0.0
    batch_size = 1
    num_imgs = len(img_list)
    step=img_prefix.split('/')[-1].split('-')[0]
    for i in tqdm(range(0, num_imgs, batch_size)):
        actual_bs = min(batch_size, num_imgs - i)
        batch_paths = img_list[i:i+actual_bs]   # 注意这里批次实际长度可能小于8
        batch_imgs = [Image.open(p) for p in batch_paths]
        batch_prompt = prompts[i:i+actual_bs]
        with torch.no_grad():
            batch_scores = calc_probs(model, processor, batch_prompt, batch_imgs)
            batch_scores = batch_scores
            for score in batch_scores:
                score_sum += score
    print(f"**{step}** Average score:", score_sum / len(img_list))