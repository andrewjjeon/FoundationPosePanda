import pybullet as p
import pybullet_data
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

physics_client = p.connect(p.GUI)

def random_camera_position(radius=0.7, min_elevation=80, max_elevation=100): #elevation angle other way, 0 is top down view
    theta = np.radians(np.random.uniform(min_elevation, max_elevation))
    phi = np.random.uniform(-np.pi/4, np.pi/4)  # generate random rotation angle around target, 0-180 deg
    # convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta) + 0.5  # Target is (0,0,0.5)
    return [x, y, z]

def extract_near_far(proj_mat):
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

# TAKE A PICTURE WITH A VIRTUAL CAMERA
def take_picture(index, w, h, view_mat, proj_mat, save_dir="./synth"):
    image_data = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view_mat,
        projectionMatrix=proj_mat,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    )
    # extract rgb, depth and seg mask
    rgb_array = np.reshape(image_data[2], (h, w, 4))[:, :, :3] #reshape rgba data into h, w, rgb
    segmentation_mask = np.reshape(image_data[4], (h, w)) #reshape segmask data into h, w

    depth_buffer = np.reshape(image_data[3], (h, w)) #reshape depth data into h, w
    near, far = extract_near_far(proj_mat)
    print(f"near: {near}, far: {far}")
    depth_corrected = far * near / (far - ((far - near) * depth_buffer))
    depth_scaled = (depth_corrected * 1000).astype(np.uint16)  # Convert to millimeters
    depth_image = Image.fromarray(depth_scaled)
    depth_image.save(f"{save_dir}/depth/{index}.png", format="PNG", bits=16)

    robot_id = p.getBodyUniqueId(0)
    hand_link_index = 8
    hand_mask_value = robot_id + ((hand_link_index + 1) << 24)
    # Generate binary mask: Set only the hand to white (255), everything else black (0)
    binary_mask = np.where(segmentation_mask == hand_mask_value, 255, 0).astype(np.uint8)
    # for whole robot mask
    # binary_mask = np.where(segmentation_mask == 0, 255, 0).astype(np.uint8)

    # Save RGB, depth and binary mask images
    rgb_image = Image.fromarray(rgb_array)
    rgb_image.save(f"{save_dir}/rgb/{index}.png")
    binary_mask_image = Image.fromarray(binary_mask)
    binary_mask_image.save(f"{save_dir}/masks/{index}.png")
    
    K = get_camera_intrinsics(w, h, proj_mat)
    print("Camera Intrinsics Matrix (K):\n", K)

    return K

def compute_hand2cam_pose(robot_id, vmat):
    # base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
    hand_link_index = 8
    link_state = p.getLinkState(robot_id, hand_link_index, computeForwardKinematics=True)
    hand_position, hand_orientation = link_state[0], link_state[1]  # Position, quaternion

    vmat = np.array(vmat).reshape(4, 4) #world2cam
    hand_position_homogeneous = np.append(hand_position, 1) # [0, 0, 0.05, 1]
    #MUST transpose because pybullet returns pmat in row-major order - different convention from numpy
    hand_position_cam_homogeneous = vmat.T @ hand_position_homogeneous
    
    hand_position_cam = hand_position_cam_homogeneous[:3]

    # Transform robot hand orientation into camera frame
    hand_orientation_world = R.from_quat(hand_orientation).as_matrix()  # quaternion to rotation matrix
    vmat_rotation = vmat.T[:3, :3]  # rotation matrix part of the world2cam transform
    hand_orientation_cam = vmat_rotation @ hand_orientation_world

    hand2cam_pose = np.eye(4)
    hand2cam_pose[:3, :3] = hand_orientation_cam
    hand2cam_pose[:3, 3] = hand_position_cam

    return hand_position_cam, hand_orientation_cam, hand2cam_pose

def project2screen(hand_position_cam, pmat, w, h):
    # convert to homogeneous coordinates
    position_homogeneous = np.append(hand_position_cam, 1)
    # apply the projection matrix to get the 2D position in homogeneous coordinates
    pmat = np.array(pmat).reshape(4, 4)
    projected_position_homogeneous = pmat.T @ position_homogeneous
    #normalize away the w_cam in homogeneous coordinates
    projected_position_norm = projected_position_homogeneous / projected_position_homogeneous[3]
    # convert back to non-homogeneous coordinates
    img_x = projected_position_norm[0]
    img_y = projected_position_norm[1]
    img_x = ((img_x + 1) / 2) * w
    img_y = ((1 - img_y) / 2) * h
    return img_x, img_y

def clamp_coords(x, y, width, height):
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    return x, y
    
def visualize_on_image(index, hand_position_cam, hand_orientation_cam, rgb_image_path, pmat, w, h):

    # Load the RGB image
    rgb_image = np.array(Image.open(rgb_image_path))
    screen_x, screen_y = project2screen(hand_position_cam, pmat, w, h)

    axis_length = 0.5
    x_axis_cam = hand_position_cam + axis_length * hand_orientation_cam[:, 0]
    y_axis_cam = hand_position_cam + axis_length * hand_orientation_cam[:, 1]
    z_axis_cam = hand_position_cam + axis_length * hand_orientation_cam[:, 2]
    x_axis_x, x_axis_y = project2screen(x_axis_cam, pmat, w, h)
    y_axis_x, y_axis_y = project2screen(y_axis_cam, pmat, w, h)
    z_axis_x, z_axis_y = project2screen(z_axis_cam, pmat, w, h)

    # Clamp to image bounds
    x_axis_x, x_axis_y = clamp_coords(x_axis_x, x_axis_y, w, h)
    y_axis_x, y_axis_y = clamp_coords(y_axis_x, y_axis_y, w, h)
    z_axis_x, z_axis_y = clamp_coords(z_axis_x, z_axis_y, w, h)



    plt.imshow(rgb_image)
    plt.scatter(screen_x, screen_y, color='red', s=50)  # s is the size of the dot
    plt.plot([screen_x, x_axis_x], [screen_y, x_axis_y], color='red', linewidth=2)  # x-axis
    plt.plot([screen_x, y_axis_x], [screen_y, y_axis_y], color='green', linewidth=2)  # y-axis
    plt.plot([screen_x, z_axis_x], [screen_y, z_axis_y], color='blue', linewidth=2)  # z-axis
    plt.savefig(f'./synth/gtviz/{index}.png')
    plt.close()


if __name__ == "__main__":
    urdf_path = "/home/andrewjjeon/FoundationPose/demo_data/panda.urdf"
    robot_id = p.loadURDF(urdf_path) #robot positioned at 0,0,0 by default

    joint_positions = [
        -0.08258942509847775,  # panda_joint1
        -0.9423399551273665,  # panda_joint2
        0.019490906304476946,  # panda_joint3
        -2.7832563950388054,  # panda_joint4
        0.03141476552378789,  # panda_joint5
        1.92817867580201,  # panda_joint6
        0.8466345183220174,  # panda_joint7
        0.040408313274383545,  # panda_finger_joint1
        0.040408313274383545  # panda_finger_joint2
    ]
    for i in range(len(joint_positions)):
        p.resetJointState(robot_id, i, joint_positions[i])

    w = 640
    h = 480
    hand2cam_poses = []

    for i in range(100):
        rand_campos = random_camera_position()
        print(f"Random camera position: {rand_campos}")

        # vmat transforms 3D coordinates from world frame to cam frame
        # x is right-left(red), y is forward-backward(green), z is above-below (blue)
        vmat = p.computeViewMatrix(cameraEyePosition=rand_campos, cameraTargetPosition=[0, 0, 0.5], cameraUpVector = [0, 0, 1])
        pmat = p.computeProjectionMatrixFOV(fov=60, aspect=(w / h), nearVal=0.001, farVal=5) 
        take_picture(i, w, h, vmat, pmat)

        vmat_reshape = np.array(vmat).reshape(4, 4)
        pmat_reshape = np.array(pmat).reshape(4, 4)
        
        hand2cam_position, hand2cam_orientation, hand2cam_pose = compute_hand2cam_pose(robot_id, vmat)
        hand2cam_poses.append(hand2cam_pose)
        print(f"hand2cam_pose:\n {hand2cam_pose}\n")

        rgb_image_path = f"./synth/rgb/{i}.png"  # Path to the original RGB image
        visualize_on_image(i, hand2cam_position, hand2cam_orientation, rgb_image_path, pmat, w, h)

    np.save("./synth/hand2cam_poses.npy", hand2cam_poses)
    input("Press Enter to close the simulation...")

p.disconnect()