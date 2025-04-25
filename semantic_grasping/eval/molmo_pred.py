from abc import ABC, abstractmethod
import re
import xml.etree.ElementTree as ElementTree
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import KDTree


DRAW_POINTS = np.array([
    [0.041, 0, 0.112],
    [0.041, 0, 0.066],
    [-0.041, 0, 0.066],
    [-0.041, 0, 0.112],
])  # in grasp frame

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

class MolmoPredictor(ABC):
    @abstractmethod
    def _pred(self, images: list[Image.Image], tasks: list[str], verbosity: int = 0) -> list[str]:
        raise NotImplementedError

    def pred_points(self, images: list[Image.Image], tasks: list[str], verbosity: int = 0) -> np.ndarray:
        """
        Args:
            images: The images of the scene.
            tasks: The tasks to predict the grasp point for.
            verbosity: The verbosity level, higher is more.
        Returns:
            The predicted points as a numpy array of shape (B, 2).
        """
        preds = self._pred(images, tasks, verbosity)

        points = []
        for pred, image in zip(preds, images):
            point = parse_point(pred, image.size)
            if point is None:
                raise ValueError(f"Failed to parse point from prediction: {pred}")
            points.append(point)
        points = np.array(points)

        if verbosity >= 1:
            print(f"Predicted points: {points}")

        if verbosity >= 3:
            for image, point in zip(images, points):
                draw = ImageDraw.Draw(image)
                r = 5
                draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill="red")

        return points

    def pred_grasp(self, images: list[Image.Image], pcs: list[np.ndarray], tasks: list[str], grasps: list[np.ndarray], cam_Ks: list[np.ndarray], verbosity: int = 0) -> list[int]:
        """
        Args:
            images: The images of the scene.
            pcs: list of (*, 3) The point clouds of the scene.
            tasks: The tasks to perform.
            grasps: list of (N, 4, 4) The grasps to choose from, in camera frame.
            cam_Ks: list of (3, 3) The camera intrinsic matrices.
        Returns:
            The indexes of the grasp to perform.
        """
        points = self.pred_points(images, tasks, verbosity=verbosity)

        grasp_idxs = []
        for i in range(len(images)):
            point = points[i]
            sample_grasps = grasps[i]
            pc = pcs[i]
            image = images[i]
            cam_K = cam_Ks[i]

            grasp_pos = sample_grasps[:, :3, 3] + sample_grasps[:, :3, 2] * 0.066

            tree = KDTree(pc[:, :3])
            _, point_idxs = tree.query(grasp_pos, k=1)
            grasp_points = pc[point_idxs, :3]

            grasp_points_2d = grasp_points @ cam_K.T
            grasp_points_2d = grasp_points_2d[:, :2] / grasp_points_2d[:, 2:3]

            dists = np.linalg.norm(grasp_points_2d - point[None], axis=1)
            grasp_idx = np.argmin(dists)
            grasp_idxs.append(grasp_idx)

            if verbosity >= 3:
                draw = ImageDraw.Draw(image)
                r = 5
                for grasp_point in grasp_points_2d:
                    draw.ellipse((grasp_point[0] - r, grasp_point[1] - r, grasp_point[0] + r, grasp_point[1] + r), fill="blue")

                grasp = sample_grasps[grasp_idx]
                draw_points = DRAW_POINTS @ grasp[:3, :3].T + grasp[:3, 3]
                draw_points_px = draw_points @ cam_K.T
                draw_points_px = draw_points_px[:, :2] / draw_points_px[:, 2:3]
                draw_points_px = draw_points_px.round().astype(int).tolist()

                for i in range(len(DRAW_POINTS)-1):
                    p0 = draw_points_px[i]
                    p1 = draw_points_px[i+1]
                    draw.line(p0 + p1, fill="red", width=2)

        return grasp_idxs
