import numpy as np
import torch

GRASP_VOLUME_SIZE = np.array([0.082, 0.01, 0.112-0.066])
GRASP_VOLUME_CENTER = np.array([0, 0, (0.066+0.112)/2])

def depth_to_pc(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    depth_mask = (depth > 0)
    uvd = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
    uvd *= np.expand_dims(depth, axis=-1)
    uvd = uvd[depth_mask]
    xyz = np.linalg.solve(cam_K, uvd.T).T
    return xyz

@torch.compile
def get_grasp_constraint_torch(grasps: torch.Tensor):
    """
    Args:
        grasp: (K, 4, 4) The grasp pose.
    Returns:
        A: (K, 6, 3) and b: (K, 6) such that Ax <= b implies that x is in the grasp volume.
    """
    min_pos, max_pos = GRASP_VOLUME_CENTER - GRASP_VOLUME_SIZE/2, GRASP_VOLUME_CENTER + GRASP_VOLUME_SIZE/2
    min_pos = torch.as_tensor(min_pos, dtype=grasps.dtype, device=grasps.device)
    max_pos = torch.as_tensor(max_pos, dtype=grasps.dtype, device=grasps.device)
    R, t = grasps[:, :3, :3], grasps[:, :3, 3]
    Rt = R.transpose(2, 1)
    Rt_times_t = (Rt @ torch.unsqueeze(t, dim=-1))[..., 0]

    A = torch.cat([Rt, -Rt], dim=1)
    b = torch.cat([max_pos + Rt_times_t, -min_pos - Rt_times_t], dim=1)

    return A, b

@torch.compile
def get_grasp_points_torch(pc: torch.Tensor, grasps: torch.Tensor):
    """
    Args:
        pc: (N, 3) The point cloud of the scene.
        grasps: (K, 4, 4) The grasp poses to get the points for.
    Returns:
        The grasp points in the point cloud, shape (K, 3).
    """
    A, b = get_grasp_constraint_torch(grasps)

    A_bc = torch.broadcast_to(torch.unsqueeze(A, dim=1), (A.shape[0], pc.shape[0], 6, 3))  # (K, N, 6, 3)
    b_bc = torch.broadcast_to(torch.unsqueeze(b, dim=1), (b.shape[0], pc.shape[0], 6))  # (K, N, 6)
    pc_bc = torch.broadcast_to(torch.unsqueeze(pc, dim=0), (A.shape[0], pc.shape[0], 3))[..., None]  # (K, N, 3, 1)

    trf_pc = torch.squeeze(A_bc @ pc_bc, dim=-1)  # (K, N, 6)
    in_grasp_mask = torch.all(trf_pc <= b_bc, dim=-1)  # (K, N), (i, j) = True iff pc[j] is in grasp i

    grasp_points = []
    for i in range(len(grasps)):
        grasp_ref_pos = grasps[i, :3, 3] + grasps[i, :3, 2] * 0.066
        if not torch.any(in_grasp_mask[i]):
            closest_idx = torch.argmin(torch.linalg.norm(pc - grasp_ref_pos[None], dim=-1))
            grasp_points.append(pc[closest_idx])
        else:
            in_grasp_points = pc[in_grasp_mask[i]]
            closest_idx = torch.argmin(torch.linalg.norm(in_grasp_points - grasp_ref_pos[None], dim=-1)).item()
            grasp_points.append(in_grasp_points[closest_idx])
    grasp_points = torch.stack(grasp_points, dim=0)
    return grasp_points

def get_grasp_points(pc: np.ndarray, grasps: np.ndarray):
    """
    Args:
        pc: (N, 3) The point cloud of the scene.
        grasps: (K, 4, 4) The grasp poses to get the points for.
    Returns:
        The grasp points, shape (K, 3).
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    pc_torch = torch.as_tensor(pc, dtype=torch.float32, device=device)
    grasps_torch = torch.as_tensor(grasps, dtype=torch.float32, device=device)
    return get_grasp_points_torch(pc_torch, grasps_torch).cpu().numpy().astype(pc.dtype)
