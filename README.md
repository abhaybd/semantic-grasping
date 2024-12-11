# Semantic Grasping

## Notes about scan data format

Mostly self-explanatory - depth and rgb in the same frame, camera intrinsics specified by `cam_info.npy`.
The image frame is `+x=right, +y=down, +z=forward`. The camera is placed at the origin.

Grasps are specified as a pose in the world frame (i.e. grasp-to-world transform), and the gripper points are in `grasp_renderer.py`.
M2T2 represents grasp frames as `+z=forward, +x=right/left`, while TaskGrasp does `+x=forward, +y=right/left`.
