import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

physics_client = p.connect(p.GUI)

def extract_near_far(proj_mat):
    """ Extract near and far clipping planes from PyBullet's projection matrix. """
    proj_mat = np.array(proj_mat).reshape(4, 4)  # Convert flat list to 4x4 matrix
    near = proj_mat[3, 2] / (proj_mat[2, 2] - 1.0)
    far = proj_mat[3, 2] / (proj_mat[2, 2] + 1.0)
    return near, far

def get_camera_intrinsics(w, h, proj_mat):
    proj_mat = np.array(proj_mat).reshape(4, 4)

    fx = w / 2 * proj_mat[0, 0]  # Focal length in x
    fy = h / 2 * proj_mat[1, 1]  # Focal length in y
    cx = w / 2 * (1 + proj_mat[0, 2])  # Principal point x
    cy = h / 2 * (1 + proj_mat[1, 2])  # Principal point y
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]])
    return K

def take_picture(w, h, view_mat, proj_mat, save_dir="./fullhand"):
    image_data = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = np.reshape(image_data[2], (h, w, 4))[:, :, :3] #reshape rgba data into h, w, rgb
    segmentation_mask = np.reshape(image_data[4], (h, w)) #reshape segmask data into h, w
    
    depth_buffer = np.reshape(image_data[3], (h, w)) #reshape depth data into h, w
    near, far = extract_near_far(proj_mat)
    print(f"near: {near}, far: {far}")
    depth_corrected = far * near / (far - ((far - near) * depth_buffer))
    depth_scaled = (depth_corrected * 1000).astype(np.uint16)  # Convert to millimeters
    depth_image = Image.fromarray(depth_scaled)
    depth_image.save(f"{save_dir}/depth/1.png", format="PNG", bits=16)

    # segmask identifies object as 0, so set 0's to white and -1s to black
    binary_mask = np.where(segmentation_mask == 0, 255, 0).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array)
    rgb_image.save(f"{save_dir}/rgb/1.png")
    binary_mask_image = Image.fromarray(binary_mask)
    binary_mask_image.save(f"{save_dir}/masks/1.png")
    
    K = get_camera_intrinsics(w, h, proj_mat)
    print("Camera Intrinsics Matrix (K):\n", K)

    return K

def compute_obj2cam_pose(obj_id, vmat):

    obj_position, obj_orientation = p.getBasePositionAndOrientation(obj_id)
    vmat = np.array(vmat).reshape(4, 4)  # World-to-camera transformation
    obj_position_homogeneous = np.append(obj_position, 1)
    obj_position_cam_homogeneous = vmat.T @ obj_position_homogeneous
    obj_position_cam = obj_position_cam_homogeneous[:3]  # Extract (x, y, z)

    obj_orientation_world = R.from_quat(obj_orientation).as_matrix()
    vmat_rotation = vmat.T[:3, :3]

    obj_orientation_cam = vmat_rotation @ obj_orientation_world

    obj2cam_pose = np.eye(4)
    obj2cam_pose[:3, :3] = obj_orientation_cam
    obj2cam_pose[:3, 3] = obj_position_cam

    return obj_position_cam, obj_orientation_cam, obj2cam_pose

def project2screen(obj_position_cam, pmat, w, h):
    # convert to homogeneous coordinates
    position_homogeneous = np.append(obj_position_cam, 1)
    # apply the projection matrix to get the 2D position in homogeneous coordinates
    pmat = np.array(pmat).reshape(4, 4)
    projected_position_homogeneous = pmat.T @ position_homogeneous
    projected_position_norm = projected_position_homogeneous / projected_position_homogeneous[3]
    # convert back to non-homogeneous coordinates
    img_x = projected_position_norm[0]
    img_y = projected_position_norm[1]
    img_x = ((img_x + 1) / 2) * w
    img_y = ((1 - img_y) / 2) * h
    return img_x, img_y


def visualize_on_image(obj_position_cam, obj_orientation_cam, rgb_image_path, pmat, w, h):

    rgb_image = np.array(Image.open(rgb_image_path))
    screen_x, screen_y = project2screen(obj_position_cam, pmat, w, h)

    axis_length = 0.05
    x_axis_cam = obj_position_cam + axis_length * obj_orientation_cam[:, 0]
    y_axis_cam = obj_position_cam + axis_length * obj_orientation_cam[:, 1]
    z_axis_cam = obj_position_cam + axis_length * obj_orientation_cam[:, 2]
    x_axis_x, x_axis_y = project2screen(x_axis_cam, pmat, w, h)
    y_axis_x, y_axis_y = project2screen(y_axis_cam, pmat, w, h)
    z_axis_x, z_axis_y = project2screen(z_axis_cam, pmat, w, h)

    plt.imshow(rgb_image)
    plt.scatter(screen_x, screen_y, color='red', s=50)  # s is the size of the dot
    plt.plot([screen_x, x_axis_x], [screen_y, x_axis_y], color='red', linewidth=2)  # x-axis
    plt.plot([screen_x, y_axis_x], [screen_y, y_axis_y], color='green', linewidth=2)  # y-axis
    plt.plot([screen_x, z_axis_x], [screen_y, z_axis_y], color='blue', linewidth=2)  # z-axis
    plt.savefig('./fullhand/gtviz.png')
    plt.close()


if __name__ == "__main__":
    # obj_vpath = "/home/andrewjjeon/FoundationPose/demo_data/meshes/visual/fullhand.obj"
    # obj_cpath = "/home/andrewjjeon/FoundationPose/demo_data/meshes/collision/fullhand.obj"
    obj_vpath = "/home/andrewjjeon/FoundationPose/demo_data/fullhand/mesh/fullhand.obj"
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=obj_vpath)
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=obj_vpath)
    obj_id = p.createMultiBody(baseMass=0,  
                               baseCollisionShapeIndex=collision_shape_id,  
                               baseVisualShapeIndex=visual_shape_id,
                               basePosition=[0, 0, 0])  

    w = 640
    h = 480
    # vmat transforms 3D coordinates from world frame to cam frame
    # x is right, y is Up, z is into-camera backward
    # cameratargetposition is simply where its pointing, not the actual position of the robot
    vmat = p.computeViewMatrix(cameraEyePosition=[0.3, 0, 0.3], cameraTargetPosition=[0, 0, 0], cameraUpVector = [0, 1, 0])
    pmat = p.computeProjectionMatrixFOV(fov=60, aspect=(w / h), nearVal=0.01, farVal=1.0)
    take_picture(w, h, vmat, pmat)

    vmat_reshape = np.array(vmat).reshape(4, 4)
    pmat_reshape = np.array(pmat).reshape(4, 4)
    
    obj2cam_position, obj2cam_orientation, obj2cam_pose = compute_obj2cam_pose(obj_id, vmat)
    print(f"obj2cam_pose:\n {obj2cam_pose}\n") #obj2cam is 100% in OpenGL camera coordinate right now
    rgb_image_path = "./fullhand/rgb/1.png"

    visualize_on_image(obj2cam_position, obj2cam_orientation, rgb_image_path, pmat, w, h)
    input("Press Enter to close the simulation...")
p.disconnect()
