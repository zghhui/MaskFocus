import requests
from PIL import Image
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

class UnifiedReward:
    def __init__(self, args):
        """Submits images to DeQA and computes a reward."""
        from requests.adapters import HTTPAdapter, Retry

        batch_size = 64
        self.url = "http://127.0.0.1:8080/v1/chat/completions"
        self.sess = requests.Session()
        retries = Retry(
            total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
        )
        self.sess.mount("http://", HTTPAdapter(max_retries=retries))

    @property
    def __name__(self):
        return 'UnifiedReward'

    def load_to_device(self, load_device):
        pass

    def parse_final_score(self, text):
        for line in text.split('\n'):
            if line.strip().startswith("Final Score:"):
                try:
                    return float(line.strip().split("Final Score:")[1].strip())
                except Exception:
                    return None
        return None

    def __call__(self, prompts, images, **kwargs):
        result = []
        max_retries = 10  # 最大重试次数
        for image, prompt_batch in zip(images, prompts):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            jpeg_bytes = buffer.getvalue()
            jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

            score = None
            output = ""
            attempt = 0

            while score is None and attempt < max_retries:
                attempt += 1
                data = {
                    "model": "UnifiedReward",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_base64}"}},
                                {"type": "text", "text": f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt_batch}]'}
                            ]
                        }
                    ],
                    "max_tokens": 2048
                }
                try:
                    response = self.sess.post(self.url, json=data, timeout=120)
                    response_data = response.json()
                    output = response_data["choices"][0]["message"]["content"]
                    score = self.parse_final_score(output)
                    score = score/5
                    if score is not None:
                        # print(f"Final Score: {score}")
                        break
                    else:
                        print(f"[Retry {attempt}] Final Score not found, retrying...")
                except Exception as e:
                    score = 0.5
                    print(f"API响应解析错误: {e}")
            result.append(score)
        return result

# 测试代码
if __name__ == "__main__":
    img = Image.new('RGB', (224, 224), color='red')
    prompts = ["a photo of a red pic"]
    images = [img]

    rewarder = UnifiedReward(args={})
    outputs = rewarder(prompts=prompts, images=images)
    print("Reward outputs:", outputs)