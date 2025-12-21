# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import VQModel
from diffusers.utils import replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from meissonic.scheduler import Scheduler
from meissonic.transformer import Transformer2DModel
import copy
import math

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> image = pipe(prompt).images[0]
        ```
"""
                        

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def print_cuda_memory(prefix=""):
    if torch.cuda.is_available():
        print(f"{prefix} CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"{prefix} CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print(f"{prefix} CUDA Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")


class Pipeline(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: Transformer2DModel 
    scheduler: Scheduler
    # tokenizer_t5: T5Tokenizer
    # text_encoder_t5: T5ForConditionalGeneration

    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: Transformer2DModel, 
        scheduler: Scheduler,
        # tokenizer_t5: T5Tokenizer,
        # text_encoder_t5: T5ForConditionalGeneration,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            # tokenizer_t5=tokenizer_t5,
            # text_encoder_t5=text_encoder_t5,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    @torch.no_grad()
    # @profile
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 48,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.IntTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),

        ref_model=None
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.

        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """
        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
            negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )

        if (prompt is None and prompt_embeds is None) or (prompt is not None and prompt_embeds is not None):
            raise ValueError("pass only one of `prompt` or `prompt_embeds`")

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        if prompt_embeds is None:
                input_ids = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77, #self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)
                # input_ids_t5 = self.tokenizer_t5(
                #     prompt,
                #     return_tensors="pt",
                #     padding="max_length",
                #     truncation=True,
                #     max_length=512,
                # ).input_ids.to(self._execution_device)
                
       
                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                # outputs_t5 = self.text_encoder_t5(input_ids_t5, decoder_input_ids = input_ids_t5 ,return_dict=True, output_hidden_states=True)
                prompt_embeds = outputs.text_embeds
                encoder_hidden_states = outputs.hidden_states[-2]
                # encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]
                
                input_ids = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77, #self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)
                # input_ids_t5 = self.tokenizer_t5(
                #     prompt,
                #     return_tensors="pt",
                #     padding="max_length",
                #     truncation=True,
                #     max_length=512,
                # ).input_ids.to(self._execution_device)

                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                # outputs_t5 = self.text_encoder_t5(input_ids_t5, decoder_input_ids = input_ids_t5 ,return_dict=True, output_hidden_states=True)
                negative_prompt_embeds = outputs.text_embeds
                negative_encoder_hidden_states = outputs.hidden_states[-2]
                # negative_encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]
                
              

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
            latents = torch.full(
                shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
            )

        self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)

        prob = torch.zeros_like(latents, dtype=torch.float)
        ref_prob = torch.zeros_like(latents, dtype=torch.float)
        
        mask_index = torch.zeros_like(latents, dtype=torch.float)
        cond_probs = torch.zeros_like(latents, dtype=torch.float)

        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        
        input_dicts = {}
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, timestep in enumerate(self.scheduler.timesteps):
            if guidance_scale > 1.0:
                model_input = torch.cat([latents] * 2)
            else:
                model_input = latents

            if height == 1024:
                img_ids = _prepare_latent_image_ids(
                    model_input.shape[0], model_input.shape[-2], model_input.shape[-1], model_input.device, model_input.dtype
                )  
            else:
                img_ids = _prepare_latent_image_ids(
                    model_input.shape[0], 2*model_input.shape[-2], 2*model_input.shape[-1], model_input.device, model_input.dtype
                )
            txt_ids = torch.zeros(
                encoder_hidden_states.shape[1], 3, device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype
            )

            input_dict = {
                "hidden_states": model_input,
                "micro_conds": micro_conds,
                "pooled_projections": prompt_embeds,
                "encoder_hidden_states": encoder_hidden_states,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "timestep": torch.tensor([timestep], device=model_input.device, dtype=torch.long),
            }   

            model_output = self.transformer(**input_dict)
            if guidance_scale > 1.0:
                uncond_logits, cond_logits = model_output.chunk(2)
                model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            
            pre_latents = latents
            cur_entropy = - (model_output.softmax(dim=-1) * torch.log(model_output.softmax(dim=-1) + 1e-12)).sum(dim=(-2,-1))#b
            
            ## dual branch sampling
            step_output = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
                ref_model_output=model_output,
                model_output_nocfg=model_output,
                ref_model_output_nocfg=model_output,
                generator=generator,
            )
            step_output_arsample = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
                ref_model_output=model_output,
                model_output_nocfg=model_output,
                ref_model_output_nocfg=model_output,
                generator=generator,
                enable_entropy_filtering=True
            )
            

            # entropy
            cur_entropy = - (model_output.softmax(dim=1) * torch.log(model_output.softmax(dim=1) + 1e-12))
            if i != 0:
                cur_entropy = cur_entropy * input_dicts["unknown_map"][-1].unsqueeze(1)
            cur_entropy = cur_entropy.sum(dim=(-3, -2,-1)) # [B]
            batch_size = latents.shape[0]
            half = batch_size // 2

            sorted_entropy, _ = torch.sort(cur_entropy, descending=True)
            threshold = sorted_entropy[half-1]

            mask = (cur_entropy >= threshold).unsqueeze(-1).unsqueeze(-1)

            latents = torch.where(
                mask,
                step_output.prev_sample,
                step_output_arsample.prev_sample
            )

            probs_step = torch.where(
                mask, 
                step_output.pred_mask_probs,
                step_output_arsample.pred_mask_probs
            )

            unknown_map_step = torch.where(
                mask,
                step_output.unknown_map,
                step_output_arsample.unknown_map
            )

            pred_original_sample = torch.where(
                mask,
                step_output.pred_original_sample,
                step_output_arsample.pred_original_sample
            )
            ## dual branch sampling
            

            prob = torch.where(unknown_map_step, prob, probs_step)
                            
            if i == 0:
                input_dicts["input_dicts"] = {
                    "micro_conds": micro_conds.chunk(2)[-1] if guidance_scale > 1.0 else micro_conds,
                    "pooled_projections": prompt_embeds.chunk(2)[-1] if guidance_scale > 1.0 else prompt_embeds,
                    "encoder_hidden_states": encoder_hidden_states.chunk(2)[-1] if guidance_scale > 1.0 else encoder_hidden_states,
                    "img_ids": img_ids,
                    "txt_ids": txt_ids         
                }
                input_dicts["timestep"] = []
                input_dicts["unknown_map"] = []
                input_dicts["inner_embeddings"] = []
                input_dicts["pred_token_map"] = []
                
            elif i == num_inference_steps-1:
                input_dicts["model_output"] = latents
            input_dicts["timestep"].append(torch.tensor([timestep], device=model_input.device, dtype=torch.long))
            input_dicts["unknown_map"].append(latents==self.scheduler.config.mask_token_id)
            input_dicts["inner_embeddings"].append(self.vqvae.quantize.get_codebook_entry(pred_original_sample, shape=(
                            batch_size,
                            height // self.vae_scale_factor,
                            width // self.vae_scale_factor,
                            self.vqvae.config.latent_channels,
                        )))
            input_dicts["pred_token_map"].append(pre_latents != latents)
                    
        if output_type == "latent":
            output = latents
        else:
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast

            if needs_upcasting:
                self.vqvae.float()

            with torch.no_grad():
                output = self.decode_with_minibatch(
                    self.vqvae, latents, batch_size, height, width,
                    self.vqvae.config.latent_channels, self.vae_scale_factor, minibatch_size=1
                )
                output = self.image_processor.postprocess(output, output_type)
                
            if needs_upcasting:
                self.vqvae.half()

        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (output, input_dicts, prob, latents)

        return ImagePipelineOutput(output)
    
    @torch.no_grad()
    def decode_with_minibatch(self, vqvae, latents, batch_size, height, width, latent_channels, vae_scale_factor, minibatch_size=1):
        """
        latents: [B, ...], float/long/cuda tensor
        Returns: Tensor of shape [B, ...],拼接所有minibatch的decode结果
        """
        outputs = []
        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            latents_mb = latents[start:end]
            shape = (
                latents_mb.shape[0],
                height // vae_scale_factor,
                width // vae_scale_factor,
                latent_channels,
            )
            out_mb = vqvae.decode(
                latents_mb,
                force_not_quantize=True,
                shape=shape,
            ).sample.clip(0, 1)
            outputs.append(out_mb)
            del out_mb
            torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0).contiguous()
    

class Pipeline_infer(DiffusionPipeline):
    image_processor: VaeImageProcessor
    vqvae: VQModel
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModelWithProjection
    transformer: Transformer2DModel 
    scheduler: Scheduler
    # tokenizer_t5: T5Tokenizer
    # text_encoder_t5: T5ForConditionalGeneration

    model_cpu_offload_seq = "text_encoder->transformer->vqvae"

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModelWithProjection,
        transformer: Transformer2DModel, 
        scheduler: Scheduler,
        # tokenizer_t5: T5Tokenizer,
        # text_encoder_t5: T5ForConditionalGeneration,
    ):
        super().__init__()

        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            # tokenizer_t5=tokenizer_t5,
            # text_encoder_t5=text_encoder_t5,
        )
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[List[str], str]] = None,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        num_inference_steps: int = 48,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.IntTensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_encoder_hidden_states: Optional[torch.Tensor] = None,
        output_type="pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        micro_conditioning_aesthetic_score: int = 6,
        micro_conditioning_crop_coord: Tuple[int, int] = (0, 0),
        temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
    ):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.transformer.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 16):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.IntTensor`, *optional*):
                Pre-generated tokens representing latent vectors in `self.vqvae`, to be used as inputs for image
                gneration. If not provided, the starting latents will be completely masked.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument. A single vector from the
                pooled and projected final hidden states.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                Pre-generated penultimate hidden states from the text encoder providing additional text conditioning.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            negative_encoder_hidden_states (`torch.Tensor`, *optional*):
                Analogous to `encoder_hidden_states` for the positive prompt.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            micro_conditioning_aesthetic_score (`int`, *optional*, defaults to 6):
                The targeted aesthetic score according to the laion aesthetic classifier. See
                https://laion.ai/blog/laion-aesthetics/ and the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            micro_conditioning_crop_coord (`Tuple[int]`, *optional*, defaults to (0, 0)):
                The targeted height, width crop coordinates. See the micro-conditioning section of
                https://arxiv.org/abs/2307.01952.
            temperature (`Union[int, Tuple[int, int], List[int]]`, *optional*, defaults to (2, 0)):
                Configures the temperature scheduler on `self.scheduler` see `Scheduler#set_timesteps`.

        Examples:

        Returns:
            [`~pipelines.pipeline_utils.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.pipeline_utils.ImagePipelineOutput`] is returned, otherwise a
                `tuple` is returned where the first element is a list with the generated images.
        """
        if (prompt_embeds is not None and encoder_hidden_states is None) or (
            prompt_embeds is None and encoder_hidden_states is not None
        ):
            raise ValueError("pass either both `prompt_embeds` and `encoder_hidden_states` or neither")

        if (negative_prompt_embeds is not None and negative_encoder_hidden_states is None) or (
            negative_prompt_embeds is None and negative_encoder_hidden_states is not None
        ):
            raise ValueError(
                "pass either both `negatve_prompt_embeds` and `negative_encoder_hidden_states` or neither"
            )

        if (prompt is None and prompt_embeds is None) or (prompt is not None and prompt_embeds is not None):
            raise ValueError("pass only one of `prompt` or `prompt_embeds`")

        if isinstance(prompt, str):
            prompt = [prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_images_per_prompt

        if height is None:
            height = self.transformer.config.sample_size * self.vae_scale_factor

        if width is None:
            width = self.transformer.config.sample_size * self.vae_scale_factor

        if prompt_embeds is None:
                input_ids = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77, #self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)
                # input_ids_t5 = self.tokenizer_t5(
                #     prompt,
                #     return_tensors="pt",
                #     padding="max_length",
                #     truncation=True,
                #     max_length=512,
                # ).input_ids.to(self._execution_device)
                
       
                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                # outputs_t5 = self.text_encoder_t5(input_ids_t5, decoder_input_ids = input_ids_t5 ,return_dict=True, output_hidden_states=True)
                prompt_embeds = outputs.text_embeds
                encoder_hidden_states = outputs.hidden_states[-2]
                # encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]

        prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1)
        encoder_hidden_states = encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                if negative_prompt is None:
                    negative_prompt = [""] * len(prompt)

                if isinstance(negative_prompt, str):
                    negative_prompt = [negative_prompt]

                input_ids = self.tokenizer(
                    negative_prompt,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=77, #self.tokenizer.model_max_length,
                ).input_ids.to(self._execution_device)
                # input_ids_t5 = self.tokenizer_t5(
                #     prompt,
                #     return_tensors="pt",
                #     padding="max_length",
                #     truncation=True,
                #     max_length=512,
                # ).input_ids.to(self._execution_device)

                outputs = self.text_encoder(input_ids, return_dict=True, output_hidden_states=True)
                # outputs_t5 = self.text_encoder_t5(input_ids_t5, decoder_input_ids = input_ids_t5 ,return_dict=True, output_hidden_states=True)
                negative_prompt_embeds = outputs.text_embeds
                negative_encoder_hidden_states = outputs.hidden_states[-2]
                # negative_encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]
                
              

            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1)
            negative_encoder_hidden_states = negative_encoder_hidden_states.repeat(num_images_per_prompt, 1, 1)

            prompt_embeds = torch.concat([negative_prompt_embeds, prompt_embeds])
            encoder_hidden_states = torch.concat([negative_encoder_hidden_states, encoder_hidden_states])

        # Note that the micro conditionings _do_ flip the order of width, height for the original size
        # and the crop coordinates. This is how it was done in the original code base
        micro_conds = torch.tensor(
            [
                width,
                height,
                micro_conditioning_crop_coord[0],
                micro_conditioning_crop_coord[1],
                micro_conditioning_aesthetic_score,
            ],
            device=self._execution_device,
            dtype=encoder_hidden_states.dtype,
        )
        micro_conds = micro_conds.unsqueeze(0)
        micro_conds = micro_conds.expand(2 * batch_size if guidance_scale > 1.0 else batch_size, -1)

        shape = (batch_size, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
            latents = torch.full(
                shape, self.scheduler.config.mask_token_id, dtype=torch.long, device=self._execution_device
            )

        self.scheduler.set_timesteps(num_inference_steps, temperature, self._execution_device)

        num_warmup_steps = len(self.scheduler.timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(self.scheduler.timesteps):
                if guidance_scale > 1.0:
                    model_input = torch.cat([latents] * 2)
                else:
                    model_input = latents
                if height == 1024: #args.resolution == 1024:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0], model_input.shape[-2],model_input.shape[-1],model_input.device,model_input.dtype)
                else:
                    img_ids = _prepare_latent_image_ids(model_input.shape[0],2*model_input.shape[-2],2*model_input.shape[-1],model_input.device,model_input.dtype)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = encoder_hidden_states.device, dtype = encoder_hidden_states.dtype)
                model_output = self.transformer(
                    hidden_states = model_input,
                    micro_conds=micro_conds,
                    pooled_projections=prompt_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    img_ids = img_ids,
                    txt_ids = txt_ids,
                    timestep = torch.tensor([timestep], device=model_input.device, dtype=torch.long),
                    # guidance = 7,
                    # cross_attention_kwargs=cross_attention_kwargs,
                )

                if guidance_scale > 1.0:
                    uncond_logits, cond_logits = model_output.chunk(2)
                    model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)

                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=latents,
                    generator=generator,
                ).prev_sample

                if i == len(self.scheduler.timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, timestep, latents)

        if output_type == "latent":
            output = latents
        else:
            needs_upcasting = self.vqvae.dtype == torch.float16 and self.vqvae.config.force_upcast

            if needs_upcasting:
                self.vqvae.float()

            output = self.vqvae.decode(
                latents,
                force_not_quantize=True,
                shape=(
                    batch_size,
                    height // self.vae_scale_factor,
                    width // self.vae_scale_factor,
                    self.vqvae.config.latent_channels,
                ),
            ).sample.clip(0, 1)
            output = self.image_processor.postprocess(output, output_type)

            if needs_upcasting:
                self.vqvae.half()

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return ImagePipelineOutput(output)