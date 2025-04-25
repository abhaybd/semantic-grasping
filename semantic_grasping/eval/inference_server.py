import base64
from io import BytesIO

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semantic_grasping.eval.molmo_local_pred import MolmoLocalPredictor

CKPT_DIR = "/weka/oe-training-default/roseh/mm_olmo/robomolmo_checkpoints/graspmolmo_cotraining_06_graspmolmo-focused_20250423_014047/latest-unsharded"

print("Loading checkpoint from", CKPT_DIR)
molmo = MolmoLocalPredictor(CKPT_DIR)
print("Done!")

app = FastAPI()

class PredictPointRequest(BaseModel):
    input_text: list[str]
    input_image: list[str]

@app.post("/api/predict_point")
async def predict_point(request: PredictPointRequest):
    if len(request.input_text) != len(request.input_image):
        raise HTTPException(status_code=400, detail="input_text and input_image must have the same length")
    if len(request.input_text) != 1:
        raise HTTPException(status_code=400, detail="Batching is not supported")

    images = []
    for image_str in request.input_image:
        images.append(Image.open(BytesIO(base64.b64decode(image_str.encode("utf-8")))).convert("RGB"))
    text = molmo._pred(images, request.input_text, verbosity=2)[0]
    return {
        "result": {
            "output": {
                "text": text
            }
        }
    }
