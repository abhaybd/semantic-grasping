import os
os.environ["MOLMO_DATA_DIR"] = "/weka/oe-training-default/mm-olmo"
os.environ["HF_DATASETS_CACHE"] = "/weka/oe-training-default/mm-olmo/hf_datasets"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
from PIL import Image

from olmo.model import Molmo
from olmo.data import build_mm_preprocessor, MMCollator

from semantic_grasping.eval.molmo_pred import MolmoPredictor


class LocalPredictor(MolmoPredictor):
    def __init__(self, ckpt_dir: str, prompt_pfx: str, device: str = "cuda"):
        self.ckpt_dir = ckpt_dir
        self.prompt_pfx = prompt_pfx
        self.model = Molmo.from_checkpoint(ckpt_dir, device=device)
        self.processor = build_mm_preprocessor(self.model.config, for_inference=True)
        self.collator = MMCollator(include_metadata=False)
        self.device = device

    def _pred(self, images: list[Image.Image], tasks: list[str], verbosity: int = 0) -> list[str]:
        samples = []
        for i, (image, task) in enumerate(zip(images, tasks)):
            s = {
                "style": "robot_control",
                "image": image,
                "prompt": f"instruction: Point to the grasp that would accomplish the following task: {task}"
            }
            samples.append(self.processor(s))
            if verbosity >= 1:
                print(f"Input {i}:", s["prompt"])
        batch = self.collator(samples)
        batch.pop("loss_masks")
        batch.pop("labels")

        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.inference_mode():
            outputs = self.model.generate(batch, max_steps=256)[0]

        ret = []
        for i in range(len(images)):
            pred = self.processor.tokenizer.decode(outputs[i, 0]).strip()
            if verbosity >= 1:
                print(f"Output {i}:", pred)
            ret.append(pred)
        return ret

class GraspMolmoLocalPredictor(LocalPredictor):
    def __init__(self, ckpt_dir: str, device: str = "cuda"):
        super().__init__(ckpt_dir, "instruction: Point to the grasp that would accomplish the following task: ", device=device)

class MolmoLocalPredictor(LocalPredictor):
    def __init__(self, device: str = "cuda"):
        ckpt_dir = "/weka/oe-training-default/roseh/molmo_pretrained_checkpoints/Molmo-7B-D-0924"
        prompt_pfx = "Point to where I should grasp to accomplish the following task: "
        super().__init__(ckpt_dir, prompt_pfx, device=device)
