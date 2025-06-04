import os
import pickle
import argparse
from glob import glob

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd
from egoscaler.data.tools import get_points_colors
from egoscaler.configs import CameraConfig as camera_cfg
from linemesh import LineMesh

# CAMERA_CONFIGS
PINHOLE_IMAGE_HEIGHT = camera_cfg.devices.aria.pinhole_image_size
PINHOLE_IMAGE_WIDTH = camera_cfg.devices.aria.pinhole_image_size
FOCAL_LEN = camera_cfg.devices.aria.focal_len
PRINCIPAL_POINT = camera_cfg.devices.aria.principal_point

def main(args):
    image = Image.open(f'./data/{args.scenario}/image.jpg')
    width, height = image.size
    depth = np.load(f'./data/{args.scenario}/depth.npy')
    
    rgbd = np.concatenate([np.array(image), depth[:, :, None]], axis=2)
    
    points, colors = get_points_colors(
        rgbd, None, width, height,
        principal_p=PRINCIPAL_POINT,
        focal_len_x=FOCAL_LEN, focal_len_y=FOCAL_LEN
    )
    
    with open(f'./data/{args.scenario}/trajectory.pkl', 'rb') as f:
        traj = pickle.load(f)
    with open(f'./data/{args.scenario}/text.txt', 'r') as f:
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
    trajectory = traj['traj']
    
    num_trajectory = trajectory.shape[0]
    sixDoF_traj = []
    bbox_mesh_geoms = []
    transforms = []
    min_radius = 0.0001
    max_radius = 0.01
    
    for i, tra in enumerate(trajectory):
        FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0])
        
        trans = tra[:3]
        pose = tra[3:]
        rotat = R.from_quat(pose).as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotat
        transform[:3, 3] = trans
        transforms.append(transform)
        
        bbox = np.concatenate([init_bbox, np.ones([len(init_bbox), 1])], axis=1)
        bbox = np.dot(transform, bbox.T).T
        bbox = bbox[:, :3]
        
        progress = (i+1) / num_trajectory  # 0.0 to 1.0
        
        radius = min_radius + progress * (max_radius - min_radius)
        radius = np.clip(radius, min_radius, max_radius)
        
        points_line = bbox.copy()
        lines_indices = np.array(lines)
        
        try:
            line_mesh = LineMesh(points_line, lines_indices, line_colors, radius=radius)
            bbox_mesh_geoms.extend(line_mesh.cylinder_segments)
        except Exception as e:
            print(f"Error creating LineMesh for trajectory index {i}: {e}")
            continue
    
        FOR.transform(transform)
        sixDoF_traj.append(FOR)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(pcd.colors) * 1.5, 0, 1))
    print(f"Narration: {narration}")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    for traj in sixDoF_traj:
        vis.add_geometry(traj)
    for bbox in bbox_mesh_geoms:
        vis.add_geometry(bbox)
        
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default=0)
    args = parser.parse_args()
    main(args)