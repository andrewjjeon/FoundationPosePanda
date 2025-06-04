import numpy as np
import os
import glob

pose_opengl_gts_path = "./synth/hand2cam_poses.npy"
pose_opengl_gts = np.load(pose_opengl_gts_path)

Topengl2cv = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
])

pose_opencv_gts = []
for pose in pose_opengl_gts:
    pose_opencv = Topengl2cv @ pose
    print(f"GT pose in opencv coordinate: \n{pose_opencv}")
    pose_opencv_gts.append(pose_opencv)

Fpose_dir = "/home/andrewjjeon/FoundationPose/debug/ob_in_cam"
Fpose_files = sorted(glob.glob(f"{Fpose_dir}/*.txt"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
Fpose_poses = []

for file in Fpose_files:
    pose_matrix = np.loadtxt(file)
    print(f"FPose pose: \n{pose_matrix}")
    Fpose_poses.append(pose_matrix)

rotation_errors = []
translation_errors = []

for i in range(len(pose_opencv_gts)):
    pose_cvgt = pose_opencv_gts[i]
    pose_fp = Fpose_poses[i]

    # Compute rotation angle
    R_gt = pose_cvgt[:3, :3]
    R_fp = pose_fp[:3, :3]
    R_rel = R_gt.T @ R_fp
    trace_R = np.trace(R_rel)
    theta = np.arccos((trace_R - 1) / 2)
    theta_deg = np.degrees(theta)

    # Compute translation error using Euclidean distance
    t_gt = pose_cvgt[:3, 3]
    t_fp = pose_fp[:3, 3]
    translation_error = np.linalg.norm(t_gt - t_fp)

    rotation_errors.append(theta_deg)
    translation_errors.append(translation_error)

avg_rotation_error = np.mean(rotation_errors)
avg_translation_error = np.mean(translation_errors)

print(f"Average Rotation Error: {avg_rotation_error:.4f}°")
print(f"Average Translation Error: {avg_translation_error:.4f}m")