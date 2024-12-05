from typing import List
import numpy as np
import re
from abc import ABC, abstractmethod

from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, GenerationConfig, AutoModelForCausalLM

import os
os.environ["HF_HOME"] = "/net/nfs2.prior/abhayd/huggingface_cache"

with open("prompt.txt", "r") as f:
    PROMPT_TEMPLATE = f.read()

def generate_prompt(task: str):
    return f"Which grasp, red or green, is more suitable for the task of \"{task}\"? Answer the question through identifying the grasp that is the most suitable, finding grasps which are definitely not suitable, and providing a ranking from most to least preferable for the grasps. First, describe where the grasps are located, then give the question response, then provide the explanation. Finally, in a single word say the color of the best grasp."

class BaseScorer(ABC):
    @abstractmethod
    def score(self, task: str, images: List[Image.Image]) -> List[str]:
        raise NotImplementedError

class MolmoScorer(BaseScorer):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

    def score(self, task: str, images: List[Image.Image]) -> List[str]:
        assert len(images) == 1 # TODO: add support for multiple images
        task_prompt = generate_prompt(task)
        imgs = [np.array(images[0])[..., :3]]
        inputs = self.processor.process(images=imgs, text=task_prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        # print("Prompt:", task_prompt)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"), tokenizer=self.processor.tokenizer
            )
        generated_tokens = output[0]
        text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(text)

class LlavaScorer(BaseScorer):
    def __init__(self):
        super().__init__()
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self.processor.tokenizer.padding_side = "left"

        self.prompt_convo_img = Image.open("prompt_convo.png")

    def _generate_convo(self):
        return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": generate_prompt("grasp a frying pan to cook")}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Observation:  \n- **Red grasp:** Located inside the frying pan, close to its center.  \n- **Green grasp:** Positioned on the handle of the frying pan, near its outer end.  \n\n**Most Suitable Grasp:**  \n**Green:** The green grasp is the most suitable for the task of \"grasp a frying pan to cook\" because it enables holding the frying pan by its handle, providing control and usability during cooking.  \n\n**Non-Suitable Grasps:**  \n**Red:** The red grasp is unsuitable because it involves grabbing the pan's interior, obstructing the cooking surface and potentially causing contamination or damage to food being prepared.  \n\n**Ranking (most to least preferable):**  \n1. Green  \n2. Red"}
            ]
        }
        ]

    def score(self, task: str, images: List[Image.Image]) -> List[str]:
        assert len(images) == 1 # TODO: add support for multiple images
        task_prompt = generate_prompt(task)
        conversations = [
            #self._generate_convo() + 
            [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task_prompt}
                ]
            }]
            for _ in range(len(images))
        ]
        prompts = [self.processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]

#        images = [self.prompt_convo_img] + images

        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        gen_ids = self.model.generate(**inputs, max_new_tokens=256)
        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        for d in decoded:
            print(re.sub(r" *(USER:|ASSISTANT:)\s+", r"\n\n\1\n", d).strip())
            print()


scorer = MolmoScorer()

task = "grasp a pan to cook something"
images = [Image.open(f"pan.png")]
scorer.score(task, images)
