import os
import sys
import argparse

from accelerate import Accelerator
import torch
from torchvision import transforms

## sys.path.insert(0, the dir of training code)

from meissonic.transformer import Transformer2DModel
from meissonic.pipeline import Pipeline_infer
from meissonic.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel

def main():
    parser = argparse.ArgumentParser(description="Stable diffusion pipeline with accelerate multi-GPU support")
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--prompts_file', type=str, required=True, help='Text file with prompts')
    parser.add_argument('--steps', type=int, default=64, help='Number of inference steps')
    parser.add_argument('--CFG', type=float, default=9, help='Guidance scale (CFG)')
    parser.add_argument('--resolution', type=int, default=1024, help='Resolution for output images')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # 只主进程创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    # 读取 prompts
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # 分配 prompts 到每个进程
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    prompts_per_proc = [
        p for i, p in enumerate(prompts) if i % world_size == rank
    ]
    index_prompt_list = list(range(len(prompts)))[rank::world_size] 
    if len(prompts_per_proc) == 0:
        print(f"No prompts assigned to rank {rank}.")
        return

    model = Transformer2DModel.from_pretrained(args.model_path)
    # model = Transformer2DModel.from_pretrained(args.base_model_path,subfolder="transformer",)
    vq_model = VQModel.from_pretrained(args.base_model_path, subfolder="vqvae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    scheduler = Scheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = Pipeline_infer(
        vq_model, tokenizer=tokenizer, text_encoder=text_encoder,
        transformer=model, scheduler=scheduler
    )
    pipe = pipe.to(device)

    negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
    steps = args.steps
    CFG = args.CFG
    resolution = args.resolution

    batch_size = 1

    for batch_start in range(0, len(prompts_per_proc), batch_size):
        batch_prompts = prompts_per_proc[batch_start:batch_start+batch_size]
        batch_indices = index_prompt_list[batch_start:batch_start+batch_size]
        batch_negative_prompts = [negative_prompt] * len(batch_prompts)
        images = pipe(
            prompt=batch_prompts,
            negative_prompt=batch_negative_prompts,
            height=resolution,
            width=resolution,
            guidance_scale=CFG,
            num_inference_steps=steps
        ).images
        for i, (prompt, index) in enumerate(zip(batch_prompts, batch_indices)):
            sanitized_prompt = prompt.replace(" ", "_")
            file_path = os.path.join(
                args.output_dir,
                f"{index}_{sanitized_prompt[:50]}.png"
            )
            images[i].save(file_path)
            print(f"[Rank {rank}] {file_path} saved.")
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("All prompts processed.")

if __name__ == "__main__":
    main()