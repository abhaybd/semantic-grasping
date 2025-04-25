import base64
import io
import json
import requests
from typing import Optional

from PIL import Image

from semantic_grasping.eval.molmo_pred import MolmoPredictor


MODEL_ENDPOINT = "https://ai2-reviz--graspmolmo-focused-cotraining-06-10k.modal.run/completion_stream"
ZERO_SHOT_MODEL_ENDPOINT = "https://ai2-reviz--uber-model-v4-synthetic.modal.run/completion_stream"

def encode_image(image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

class MolmoWebPredictor(MolmoPredictor):
    def __init__(self, model_endpoint: str, headers: dict, prompt_prefix: Optional[str] = None):
        self.model_endpoint = model_endpoint
        self.headers = headers
        if prompt_prefix is None:
            self.prompt_prefix = "robot_control: instruction: Point to the grasp that would accomplish the following task: "
        else:
            self.prompt_prefix = prompt_prefix

    def _send_request(self, payload: dict):
        headers = {"Content-Type": "application/json", **self.headers}
        response = requests.post(
            self.model_endpoint,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )
        return response

    def _pred(self, images: list[Image.Image], tasks: list[str], verbosity: int = 0) -> list[str]:
        ret = []
        for image, task in zip(images, tasks):
            image_enc = encode_image(image)

            payload = {
                "input_text": [f"{self.prompt_prefix}{task}"],
                "input_image": [image_enc]
            }
            try:
                response = self._send_request(payload)
                if verbosity >= 1:
                    print(f"Molmo API Response Status Code: {response.status_code}")

                # Check if response is valid
                if response.status_code != 200:
                    raise ValueError(f"[ERROR] Molmo API failed: {response.text}")

                # Process streaming response
                response_text = ""
                for chunk in response.iter_lines():
                    if chunk:
                        response_text += json.loads(chunk)["result"]["output"]["text"]
                if verbosity >= 1:
                    print("Molmo API Response:", response_text)
            except Exception as e:
                raise ValueError(f"[ERROR] Failed to query Molmo API: {str(e)}")
            ret.append(response_text)
        return ret

class ZeroShotMolmoModal(MolmoWebPredictor):
    def __init__(self, token: str):
        super().__init__(ZERO_SHOT_MODEL_ENDPOINT, {"Authorization": f"Bearer {token}"})

class GraspMolmoModal(MolmoWebPredictor):
    def __init__(self):
        super().__init__(MODEL_ENDPOINT, {})

class GraspMolmoBeaker(MolmoWebPredictor):
    def __init__(self, endpoint: str):
        super().__init__(endpoint, {}, prompt_prefix="")
