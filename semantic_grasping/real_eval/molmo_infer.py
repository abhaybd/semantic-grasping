import base64
import io
import json
import requests
from abc import ABC
import re
import xml.etree.ElementTree as ElementTree
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import KDTree

MODEL_ENDPOINT = "https://ai2-reviz--graspmolmo-focused-cotraining-06-10k.modal.run/completion_stream"
ZERO_SHOT_MODEL_ENDPOINT = "https://ai2-reviz--uber-model-v4-synthetic.modal.run/completion_stream"

DRAW_POINTS = np.array([
    [0.041, 0, 0.112],
    [0.041, 0, 0.066],
    [-0.041, 0, 0.066],
    [-0.041, 0, 0.112],
])

def encode_image(image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

def parse_point(pred: str, image_size: Optional[tuple[int, int]] = None):
    """
    Args:
        pred: The prediction string from the model.
        image_size: The size of the image, (width, height). If provided, return in pixels, otherwise return in normalized coordinates.
    Returns:
        The predicted point as a numpy array of shape (2,).
    """
    point_xmls = re.findall(r'<point.*?</point>', pred, re.DOTALL)
    if len(point_xmls) == 0:
        print(f"Invalid prediction: {pred}")
        return None
    point_xml = point_xmls[0]
    try:
        point_elem = ElementTree.fromstring(point_xml)
        
        if point_elem is not None:
            x = float(point_elem.get('x'))
            y = float(point_elem.get('y'))
            ret = np.array([x, y])
            if image_size is not None:
                ret = ret / 100 * np.array(image_size)
            return ret
        else:
            print("No point element found in XML")
    except ElementTree.ParseError as e:
        print(f"Failed to parse XML: {e}")
    return None

class MolmoInfer(ABC):
    def __init__(self, model_endpoint: str, headers: dict):
        self.model_endpoint = model_endpoint
        self.headers = headers

    def _send_request(self, payload: dict) -> str:
        headers = {"Content-Type": "application/json", **self.headers}
        response = requests.post(
            self.model_endpoint,
            headers=headers,
            data=json.dumps(payload),
            stream=True
        )
        return response

    def pred_point(self, image: Image.Image, task: str, verbosity: int = 0) -> Optional[np.ndarray]:
        image_enc = encode_image(image)

        payload = {
            "input_text": [f"robot_control: instruction: Point to the grasp that would accomplish the following task: {task}"],
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

        point = parse_point(response_text, image.size)
        if verbosity >= 1:
            print("Predicted point:", point)
        if verbosity >= 3:
            draw = ImageDraw.Draw(image)
            r = 5
            draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill="green")
        return point

    def pred_grasp(self, image: Image.Image, pc: np.ndarray, task: str, grasps: np.ndarray, cam_K: np.ndarray, verbosity: int = 0) -> int:
        """
        Args:
            image: The image of the scene.
            pc: (N, 3) The point cloud of the scene.
            task: The task to perform.
            grasps: (N, 4, 4) The grasps to choose from, in camera frame.
            cam_K: (3, 3) The camera intrinsic matrix.
        Returns:
            The index of the grasp to perform.
        """
        point = self.pred_point(image, task, verbosity=verbosity)
        if point is None:
            return point

        grasp_pos = grasps[:, :3, 3] + grasps[:, :3, 2] * 0.066

        tree = KDTree(pc[:, :3])
        _, point_idxs = tree.query(grasp_pos, k=1)
        grasp_points = pc[point_idxs, :3]

        grasp_points_2d = grasp_points @ cam_K.T
        grasp_points_2d = grasp_points_2d[:, :2] / grasp_points_2d[:, 2:3]

        dists = np.linalg.norm(grasp_points_2d - point[None], axis=1)
        grasp_idx = np.argmin(dists)

        if verbosity >= 3:
            draw = ImageDraw.Draw(image)
            r = 5
            for grasp_point in grasp_points_2d:
                draw.ellipse((grasp_point[0] - r, grasp_point[1] - r, grasp_point[0] + r, grasp_point[1] + r), fill="blue")

            grasp = grasps[grasp_idx]
            draw_points = DRAW_POINTS @ grasp[:3, :3].T + grasp[:3, 3]
            draw_points_px = draw_points @ cam_K.T
            draw_points_px = draw_points_px[:, :2] / draw_points_px[:, 2:3]
            draw_points_px = draw_points_px.round().astype(int).tolist()

            for i in range(len(DRAW_POINTS)-1):
                p0 = draw_points_px[i]
                p1 = draw_points_px[i+1]
                draw.line(p0 + p1, fill="red", width=2)

        return grasp_idx

class ZeroShotMolmo(MolmoInfer):
    def __init__(self, token: str):
        super().__init__(ZERO_SHOT_MODEL_ENDPOINT, {"Authorization": f"Bearer {token}"})

class GraspMolmo(MolmoInfer):
    def __init__(self):
        super().__init__(MODEL_ENDPOINT, {})
