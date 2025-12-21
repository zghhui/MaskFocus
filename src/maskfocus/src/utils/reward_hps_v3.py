import os
import os
import torch
import requests
from PIL import Image
from hpsv3 import HPSv3RewardInferencer
import tempfile
import os
    
class HPSv3:
    def __init__(self, args):
        self.ckpt_path = f"{args.hps_v3_ckpt_hub}/HPSv3.safetensors"
        self.config_path = f"{args.hps_v3_ckpt_hub}/hpsv3/config/HPSv3_7B.yaml"
        
    @property
    def __name__(self):
        return 'HPSv3'
    
    def load_to_device(self, load_device):
        self.inferencer = HPSv3RewardInferencer(self.config_path, self.ckpt_path, load_device)
    

    def pil_to_temp_path(self, pil_image, suffix=".png"):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        pil_image.save(tmp_file.name)
        tmp_file.close()
        return tmp_file.name

    def __call__(self, prompts, images, **kwargs):
        device = list(self.model.parameters())[0].device
        result = []
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            temp_path = self.pil_to_temp_path(image)
            try:
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.inferencer.reward([temp_path], [prompt])
                result.append(outputs[0].item()/5)
            finally:
                os.remove(temp_path)
        return result