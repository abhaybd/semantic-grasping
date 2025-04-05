import numpy as np
from tqdm import tqdm as tqdm_

def backproject(cam_K: np.ndarray, depth: np.ndarray):
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    xyz = uvd @ np.expand_dims(np.linalg.inv(cam_K).T, axis=0)
    return xyz

class tqdm(tqdm_):
    def __init__(self, *args, **kwargs):
        kwargs["bar_format"] = "{l_bar}{bar}{r_bar}\n"
        super().__init__(*args, **kwargs)
