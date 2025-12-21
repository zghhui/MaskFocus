# Copyright 2025 The HuggingFace Team. All rights reserved.
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

'''
Two Forward Passes
'''

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image

import numpy as np
import torch
from torch import nn
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from utils.reward_hps import HPSv2
# from utils.reward_hps_v3 import HPSv3
from utils.reward_clip import Clip 
from utils.reward_unifiedreward import UnifiedReward
from utils.reward_geneval import Geneval


# from utils.reward_git import GIT
# from utils.reward_gdino import GDino
# from utils.reward_orm import ORM

import torch
from meissonic.transformer import Transformer2DModel
from meissonic.pipeline import Pipeline
from meissonic.scheduler import Scheduler

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)
from transformers.training_args import OptimizerNames
from diffusers import VQModel

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

from transformers.trainer import *
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


from torch.utils.data import Sampler
from typing import Any, Callable, Optional, Sized, Union
class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count
        
class MaskFoucsTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        #### Meissonic model ####
        model_id = model
        self.model_id = model_id
        model = Transformer2DModel.from_pretrained(model_id,subfolder="transformer",)
        vq_model = VQModel.from_pretrained(model_id, subfolder="vqvae", )
        text_encoder = CLIPTextModelWithProjection.from_pretrained(   #using original text enc for stable sampling
                    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                )
        tokenizer = CLIPTokenizer.from_pretrained(model_id,subfolder="tokenizer",)
        self.scheduler = Scheduler.from_pretrained(model_id,subfolder="scheduler",)
        pipe=Pipeline(vq_model, tokenizer=tokenizer,text_encoder=text_encoder,transformer=model,scheduler=self.scheduler)

        # Reference model
        if is_deepspeed_zero3_enabled() and args.beta != 0:
            self.ref_model = Transformer2DModel.from_pretrained(model_id,subfolder="transformer",)
        elif peft_config is None and args.beta != 0:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None        

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'hpsv2' in reward_func:
                reward_funcs[i] = HPSv2(args)
            elif isinstance(reward_func, str) and 'unifiedreward' in reward_func:
                reward_funcs[i] = UnifiedReward(args)
            elif isinstance(reward_func, str) and 'hpsv3' in reward_func:
                reward_funcs[i] = HPSv3(args)
            elif isinstance(reward_func, str) and 'clip' in reward_func:
                reward_funcs[i] = Clip(args)                
            elif isinstance(reward_func, str) and 'git' in reward_func:
                reward_funcs[i] = GIT(args)
            elif isinstance(reward_func, str) and 'gdino' in reward_func:
                reward_funcs[i] = GDino(args)
            elif isinstance(reward_func, str) and 'orm' in reward_func:
                reward_funcs[i] = ORM(args)
            elif isinstance(reward_func, str) and 'geneval' in reward_func:
                reward_funcs[i] = Geneval(args)
            else:
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.new_generations_image = args.new_generations_image

        self.beta = args.beta


        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        
        #### Meissonic model ####
        self._pipe = pipe.to(self.accelerator.device)
        #### other model ####
        if self.beta != 0:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            self.ref_model = None
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, HPSv2) or isinstance(reward_func, Clip) or isinstance(reward_func, UnifiedReward) or isinstance(reward_func, HPSv3) or isinstance(reward_func, GDino) or isinstance(reward_func, GIT):
                reward_func.load_to_device(self.accelerator.device)
            elif isinstance(reward_func, Geneval):
                pass
            elif isinstance(reward_func, ORM):
                reward_func.load_to_device(self.accelerator.device)
                reward_func.accelerator = self.accelerator
                if self.is_deepspeed_enabled:   
                    reward_func.model = prepare_deepspeed(reward_func.model, self.accelerator)
                else:
                    reward_func.model = self.accelerator.prepare_model(reward_func.model, evaluation_mode=True)
        
        self.cfg = args.cfg_weight
        self.resolution = args.resolution
        self.gen_steps = args.gen_steps
        self.cirtical_steps = args.cirtical_steps
        self.image_gen_temperature = 1
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.max_textcot_length = args.max_textcot_length

        # image loss is moved to grpo_trainer_two_forward_imageloss
        # assert not self.image_loss
        
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_embeds, text_ids, img_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []

            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        # here, we only compute either text or image loss, so ids of other one could be omitted
        if img_ids is not None:
            # compute logits for image tokens
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            # (text input id, image start token, image input id)
            # text_ids: text input id + image start token
            # img_ids: img_id (image token)
            image_logits = model.gen_head(last_hidden_states[:, -(img_ids.size(1)+1):, :]) # image prediction
            
            img_input_ids = torch.cat([img_ids.new_zeros(img_ids.size(0), 1), img_ids], dim=1) # cat a random one here, since it is not used in the loss calculation
            per_token_logps_img = _get_per_token_logps_part(image_logits, img_input_ids) # only calculate image loss
            return torch.cat([
                per_token_logps_img.new_zeros(
                    (per_token_logps_img.size(0), input_embeds.size(1) - per_token_logps_img.size(1) - 1)
                ), # the return length should be the input length minus 1 (the last token does not need predict)
                per_token_logps_img
            ], 
            dim=1)
        else: # only calculate text ids
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            text_logits = model.language_model.lm_head(last_hidden_states) 
            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids) 
            return per_token_logps_text


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def mask_get_per_token_logps(self, model, all_model_output, guidance_scale):
        def move_to_device(x, device):
            return x.to(device) if hasattr(x, "to") else x
        for i, timestep in enumerate(self.scheduler.timesteps):
            model_input, micro_conds, prompt_embeds, encoder_hidden_states, img_ids, txt_ids, timestep, guidance_scale, generator = all_model_output[i]
                        
            model_output = model(
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

            step_output = self.scheduler.step(
                model_output=model_output,
                timestep=timestep,
                sample=latents,
                generator=generator,
            )
            latents, probs_step, unknown_map_step = step_output.prev_sample, step_output.pred_mask_probs, step_output.unknown_map
            prob = torch.where(unknown_map_step, probs_step, prob)
        return prob
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Geneval-style data
        prompts = [x["raw_prompt"] for x in inputs]
        prompts = [p for p in prompts for _ in range(self.num_generations)]
        if self.geneval_style:
            meta_datas = [x["meta_datas"] for x in inputs]
            expanded_meta_datas = []
            for meta in meta_datas:
                expanded_meta_datas.extend([meta] * self.num_generations)
            
        current_step = self.state.global_step
        device = self.accelerator.device

        negative_prompt = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
        generator = torch.Generator(device=device)
        autocast = self.accelerator.autocast
        with autocast():
            with torch.no_grad():
                images, input_dicts, logp_infer, index_infer = self._pipe(
                    prompt=prompts, 
                    negative_prompt=[negative_prompt] * len(prompts),
                    height=self.resolution,
                    width=self.resolution,
                    guidance_scale=self.cfg,
                    generator=generator,
                    num_inference_steps=self.gen_steps,
                    # ref_model=self.ref_model,
                    return_dict=False
                    )

        if current_step % 10 == 0:
            save_dir = os.path.join(self.img_save_dir, f"step-{current_step}")
            if self.accelerator.is_main_process:
                os.makedirs(save_dir, exist_ok=True)
            self.accelerator.wait_for_everyone()
            for index, image in enumerate(images):
                global_rank = self.accelerator.process_index
                safe_prompt = prompts[index].replace(" ", "_")
                safe_prompt = safe_prompt.replace("/", "_")
                image.save(os.path.join(save_dir, f"index_{index}_gpu_{global_rank}_{safe_prompt[:50]}.png"))


        # Compute the rewards

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            if isinstance(reward_func, Geneval):
                output_reward_func = reward_func(prompts=prompts, images=images, geneval_meta_data=expanded_meta_datas, **reward_kwargs)
            else:
                output_reward_func = reward_func(prompts=prompts, images=images, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        
        prompt_per_num = self.num_generations
        mean_grouped_rewards = rewards.view(-1, prompt_per_num).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, prompt_per_num).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(prompt_per_num, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(prompt_per_num, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        num_inference_steps = len(self._pipe.scheduler.timesteps)
        
        inner_embeddings = input_dicts["inner_embeddings"]
        gt = inner_embeddings[-1]
        inner_embeddings = inner_embeddings[:-1]
        bs = gt.shape[0]
        inner_similarity = [
            torch.nn.functional.cosine_similarity(emb.view(bs, -1), gt.view(bs, -1), dim=1).mean().item()
            for emb in inner_embeddings
        ]
        diff_simi = [abs(inner_similarity[i+1] - inner_similarity[i]) for i in range(len(inner_similarity) - 1)]

        # 第二步：选择 top 6 的索引
        diff_simi_tensor = torch.tensor(diff_simi)
        # topk 返回的是值和索引
        topk = torch.topk(diff_simi_tensor, k=self.cirtical_steps)
        num_inference_steps_list = sorted(topk.indices.tolist())

        print(num_inference_steps_list)
        
        last_latent = input_dicts["model_output"]
        
        mean_kl_list = []
        loss_list = []
        ## policy old
        old_per_token_logp_list = []
        
        apply_mask_list = []
        pred_token_map_list = []
        for i in num_inference_steps_list:
            with autocast():
                with torch.no_grad():
                    input_dict = input_dicts["input_dicts"]
                    unknown_map = input_dicts["unknown_map"][i]

                    # create_apply_mask
                    input_dict["hidden_states"], create_mask = self.create_apply_mask(last_latent, unknown_map, mask_token_flag=self._pipe.scheduler.config.mask_token_id)
                    apply_mask_list.append(create_mask)
                    pred_token_map_list.append(input_dicts["pred_token_map"][i])
                    input_dict["timestep"] = input_dicts["timestep"][i]
                    
                    # policy model logp
                    old_per_token_logp, _ = self.compute_logp(self._pipe.transformer, input_dict, index_infer)                        
                    old_per_token_logp_list.append(old_per_token_logp)
                    
        for index, i in enumerate(num_inference_steps_list):
            with autocast():
                input_dict = input_dicts["input_dicts"]
                unknown_map = input_dicts["unknown_map"][i]
                pred_token_map = input_dicts["pred_token_map"][i]
                
                input_dict["hidden_states"] = self.apply_mask(last_latent, apply_mask_list[index], mask_token_flag=self._pipe.scheduler.config.mask_token_id)
                
                input_dict["timestep"] = input_dicts["timestep"][i]
                inner_embeddings = input_dicts["inner_embeddings"]
                
                # policy logp
                per_token_logp, per_entropy = self.compute_logp(self._pipe.transformer, input_dict, index_infer)
                # ref logp, 无梯度
                with torch.no_grad():
                    ref_per_token_logp, ref_per_entropy = self.compute_logp(self.ref_model, input_dict, index_infer)
                
                # per_entropy = (per_entropy * per_entropy.float()).sum(dim=(-1, -2)) / unknown_map.float().sum(dim=(-1, -2))
                # ref_per_entropy = (ref_per_entropy * unknown_map.float()).sum(dim=(-1, -2)) / unknown_map.float().sum(dim=(-1, -2))
            
                mask_token_num = apply_mask_list[index].float().sum(dim=(-1, -2))
                pre_token_num = pred_token_map_list[index].float().sum(dim=(-1, -2))
                pred_token_w = (pre_token_num / mask_token_num).unsqueeze(1).unsqueeze(1)
                
                per_token_loss = torch.exp(per_token_logp - old_per_token_logp_list[index]) * advantages.unsqueeze(1).unsqueeze(1)
                
                per_token_kl = torch.exp(ref_per_token_logp - per_token_logp) - (ref_per_token_logp - per_token_logp) - 1
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
                
                # mask后聚合loss
                completion_mask = (apply_mask_list[index]).float()
                # print(completion_mask.sum())
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
                # print(loss)
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                
                if self.accelerator.is_main_process:
                    print(f"loss: {loss.item()}, mean_kl: {mean_kl.item()}")                
                
                mean_kl_list.append(mean_kl)
                loss_list.append(loss)
                yield loss
        
        # Log the metrics 
        mean_kl_sum = 0 
        mean_loss_sum = 0 
        for mean_kl, loss in zip(mean_kl_list, loss_list):
            mean_kl_sum += self.accelerator.gather_for_metrics(mean_kl).nanmean().item()
            mean_loss_sum+= self.accelerator.gather_for_metrics(loss).nanmean().item()
        self._metrics[f"kl"].append(mean_kl_sum / len(mean_kl_list))
        self._metrics[f"loss"].append(mean_loss_sum / len(loss_list))
           
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
 
        return loss

    def apply_mask(self, last_latent, mask, mask_token_flag):
        return last_latent.masked_fill(mask, mask_token_flag)
    
    def create_apply_mask(self, last_latent, mask, mask_token_flag):
        mask_num = mask.sum(dim=(-1, -2))[0].item()
        b = last_latent.shape[0]
        h, w = last_latent.shape[1:3]
        device = last_latent.device

        size = h * w
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        num = mask_num.item() if isinstance(mask_num, torch.Tensor) else mask_num
        idx = torch.randperm(size, device=device)[:num]
        mask[idx] = True
        mask = mask.view(h, w)  # shape [h, w]

        mask = mask.unsqueeze(0).repeat(b, 1, 1)

        return last_latent.masked_fill(mask, mask_token_flag), mask

    def compute_logp(self, transformer, input_dict_step, index_infer):
     
        model_output = transformer(**input_dict_step)
        model_output = model_output.permute(0, 2, 3, 1)
        probs_log = model_output.float().log_softmax(dim=-1) 
        probs = model_output.detach().float().softmax(dim=-1)
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1) 
        if torch.isnan(entropy).any():
            entropy = torch.zeros_like(entropy, device=entropy.device)
        probs_log = torch.gather(probs_log, -1, index_infer.unsqueeze(-1)).squeeze(-1)
        return probs_log, entropy

    def get_z_mean_similarity(self, select_embedding, gt):
        b, z, h, w = select_embedding.shape
        emb = select_embedding.permute(0, 2, 3, 1)  # [b, h, w, z]
        emb_flat = emb.reshape(-1, z)
        
        gt_emb = gt.permute(0, 2, 3, 1)  # [b, h, w, z]
        gt_emb_flat = gt_emb.reshape(-1, z)
        
        sim = torch.nn.functional.cosine_similarity(
            emb_flat, gt_emb_flat, dim=1
        )
        # [b, h, w]
        mean_sim = sim.view(b, h, w)
        return mean_sim
    
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        losses = []
        kwargs = {}
        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        with self.compute_loss_context_manager():
            for loss in self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch):
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    # Finally we need to normalize the loss for reporting
                    if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                        loss = loss / self.args.gradient_accumulation_steps

                    # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                    # https://github.com/huggingface/transformers/pull/35808
                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                        kwargs["scale_wrt_gas"] = False

                    self.accelerator.backward(loss, **kwargs)
                losses.append(loss.detach())

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        return sum(losses) / len(losses)
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        import shutil
        from transformers.utils import is_peft_available, is_safetensors_available
        from transformers.modeling_utils import PreTrainedModel
        if is_peft_available():
            from peft import PeftModel
        if is_safetensors_available():
            import safetensors.torch
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                print("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                shutil.copy(f"{self.model_id}/transformer/config.json", f"{output_dir}/config.json")
                SAFE_WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            print("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        TRAINING_ARGS_NAME = "training_args.bin"
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))