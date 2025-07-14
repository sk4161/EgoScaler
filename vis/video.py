import os
import pickle
from glob import glob

import numpy as np
import open3d as o3d
from PIL import Image
from moviepy.editor import ImageSequenceClip
from scipy.spatial.transform import Rotation as R

from linemesh import LineMesh
from egoscaler.data.tools import get_points_colors
from egoscaler.configs import CameraConfig as camera_cfg


# CAMERA_CONFIGS
PINHOLE_IMAGE_HEIGHT = camera_cfg.devices.aria.pinhole_image_size
PINHOLE_IMAGE_WIDTH = camera_cfg.devices.aria.pinhole_image_size
FOCAL_LEN = camera_cfg.devices.aria.focal_len
PRINCIPAL_POINT = camera_cfg.devices.aria.principal_point

def create_lineset(num):
    lines = []
    for i in range(num-1):
        line = np.array([i, i+1])  # 0 to 9, then 1 to 10 as pairs
        lines.append(line)
    return np.array(lines)

def compute_bbox_rotation_matrix(bbox_points):
    """
    Compute the rotation matrix of the bounding box using PCA.
    
    Parameters:
        bbox_points (np.ndarray): Array of shape (8, 3) representing the 8 corners of the bbox.
    
    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    # Compute the centroid
    centroid = np.mean(bbox_points, axis=0)
    
    # Center the points
    centered_points = bbox_points - centroid
    
    # Perform PCA
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by descending eigenvalues
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1
    
    return eigenvectors

def main():
    # Load data
    image = Image.open(f'./assets/demo/image.jpg')
    depth = np.load(f'./assets/demo/depth.npy')
    width, height = image.size

    rgbd = np.concatenate([np.array(image), depth[:, :, None]], axis=2)

    points, colors = get_points_colors(
        rgbd, None, width, height,
        principal_p=PRINCIPAL_POINT,
        focal_len_x=FOCAL_LEN, focal_len_y=FOCAL_LEN
    )

    with open(f'./assets/demo/trajectory.pkl', 'rb') as f:
        traj = pickle.load(f)
    with open(f'./assets/demo/text.txt', 'r') as f:
        narration = f.read()

    lines = [
        [0, 1], [0, 2], [0, 3],
        [4, 5], [4, 6], [4, 7],
        [5, 2], [5, 3], [6, 1],
        [6, 3], [7, 2], [7, 1]
    ]
    line_colors = [
        [0, 0, 1], [0, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 0], [1, 0, 0],
        [0, 1, 0], [0, 1, 0], [1, 0, 0],
        [0, 0, 1], [0, 0, 1], [1, 0, 0]
    ]

    # Align to zero
    init_bbox = traj['init_bbox']
    init_bbox_center = np.mean(init_bbox, axis=0)
    init_bbox -= init_bbox_center

    init_bbox_rot = compute_bbox_rotation_matrix(init_bbox)
    trajectory = traj['traj']

    # Create cache_imgs directory if not exists
    os.makedirs('./cache_imgs', exist_ok=True)

    # Initialize Open3D Visualizer
    H, W = 1100, 1400
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, window_name='PCD')
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])

    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, 1.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, -1.0, 0.0])

    vis.poll_events()
    vis.update_renderer()

    base_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform(base_transform)
    vis.add_geometry(pcd)

    num_trajectory = trajectory.shape[0]
    num_frames = num_trajectory
    radius = 0.1
    angle_increment = 2 * np.pi / num_frames  # Radians per frame

    for i, tra in enumerate(trajectory):
        FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        trans = tra[:3]
        pose = tra[3:]
        rotat = R.from_quat(pose).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotat
        transform[:3, 3] = trans

        bbox = np.concatenate([init_bbox, np.ones([len(init_bbox), 1])], axis=1)
        bbox = np.dot(transform, bbox.T).T
        bbox = np.dot(base_transform, bbox.T).T
        bbox = bbox[:, :3]

        points_line = bbox.copy()
        lines_indices = np.array(lines)

        try:
            line_mesh = LineMesh(points_line, lines_indices, line_colors, radius=0.01)
            bbox_mesh_geoms = line_mesh.cylinder_segments
        except Exception as e:
            print(f"Error creating LineMesh for trajectory index {i}: {e}")
            bbox_mesh_geoms = []

        FOR.rotate(init_bbox_rot, center=(0, 0, 0))
        FOR.transform(transform)
        FOR.transform(base_transform)

        vis.add_geometry(FOR)
        for bbox_geom in bbox_mesh_geoms:
            vis.add_geometry(bbox_geom)

        angle = i * angle_increment
        new_camera_x = radius * np.cos(angle)
        new_camera_y = radius * np.sin(angle)
        ctr.set_front([0.0 + new_camera_x, 0.0 + new_camera_y, 1.0])
        ctr.set_lookat([0.0 + new_camera_x, 0.0 + new_camera_y, 0.0])
        ctr.set_up([0.0, 1.0, 0.0])
        ctr.set_zoom(0.3)

        vis.poll_events()
        vis.update_renderer()

        image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        pil_image.save(f'./cache_imgs/{i}.jpg')

        vis.remove_geometry(FOR)
        for bbox_geom in bbox_mesh_geoms:
            vis.remove_geometry(bbox_geom)

        print(f"Processed frame {i+1}/{num_trajectory}")

    vis.destroy_window()

    # Define frames per second
    fps = max(1, int(num_frames / 4))  # Avoid fps=0

    # Create a MoviePy ImageSequenceClip
    clip = ImageSequenceClip(
        sorted(
            glob("./cache_imgs/*.jpg"),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        ),
        fps=fps
    )

    # Write the video file
    output_filename = 'visualization_video.mp4'
    clip.write_videofile(output_filename)

    for _ in glob("./cache_imgs/*.jpg"):
        os.remove(_)

    print(narration)

if __name__ == "__main__":
    main()