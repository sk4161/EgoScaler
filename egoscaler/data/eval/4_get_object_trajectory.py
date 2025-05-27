import os
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.headsets import Headset
from tqdm import tqdm
from glob import glob
from projectaria_tools.core import data_provider, calibration
from depth import DepthAnything
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import torch
from copy import deepcopy
from egoscaler.configs import CameraConfig as camera_cfg
from egoscaler.data.tools import get_points_colors

def main(args):
    object_library_path = os.path.join(args.root_dir, "assets")
    object_library = load_object_library(object_library_folderpath=object_library_path)
    sequence_paths = glob(f'{args.root_dir}/P*')
    dataset_name = 'hot3d'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything(args, device)

    for sequence_path in sequence_paths:
        try:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=sequence_path,
                object_library=object_library,
                mano_hand_model=None,
            )
        except Exception as e:
            print(e)
            continue
        
        device_pose_provider = hot3d_data_provider.device_pose_data_provider   
        object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        
        # since Meta Quest not contain rgb data, just skipping
        if hot3d_data_provider.get_device_type() != Headset.Aria:
            continue
        
        video_uid = sequence_path.split('/')[-1]
        provider = data_provider.create_vrs_data_provider(f"{sequence_path}/recording.vrs")
        
        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)
        device_calibration = provider.get_device_calibration()
        rgb_camera_calibration = device_calibration.get_camera_calib(camera_label)
        T_device_from_camera = rgb_camera_calibration.get_transform_device_camera()
        pinhole = calibration.get_linear_camera_calibration(
            camera_cfg.devices.aria.pinhole_image_size, 
            camera_cfg.devices.aria.pinhole_image_size, 
            camera_cfg.devices.aria.focal_len, 
            camera_label,
            T_device_from_camera
        )
        pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole)
        R_intri = pinhole_cw90.get_transform_device_camera().rotation().to_matrix()
        T_intri = pinhole_cw90.get_transform_device_camera().translation()

        time_domain = TimeDomain.TIME_CODE
        option = TimeQueryOptions.CLOSEST
        timestamps: list = provider.get_timestamps_ns(stream_id, time_domain)
        video_start_ns, video_end_ns = timestamps[0], timestamps[-1]
        
        # split whole video into 4 sec segmments
        segments = np.arange(video_start_ns, video_end_ns, 2*camera_cfg.time_window*1e+9)
        
        for start_ns, end_ns in zip(segments, segments[1:]):
            file_name = str(int((end_ns+start_ns) / 2))
            
            if os.path.exists(f"{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json"):
                with open(f"{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json", 'r') as f:
                    data = json.load(f)
            else:
                continue
            
            if 'start_sec' not in data or data['start_sec'] is None:
                continue
            
            sampling_rate = 1 / camera_cfg.fps
            original_duration = np.arange(start_ns, end_ns, 1e+9 * sampling_rate).astype(int)
            
            start_sec = data['start_sec']
            end_sec = data['end_sec']
            start_nsec = int(start_sec * 1e+9)
            end_nsec = int(end_sec * 1e+9)
            
            start_index = np.where(original_duration == start_nsec)[0]
            end_index = np.where(original_duration == end_nsec)[0]
            
            duration = original_duration[start_index[0]:end_index[0] + 1]
            
            #if os.path.exists(f"{args.save_dir}/trajs/{dataset_name}/{video_uid}/{file_name}.pkl"):
            #    continue
            
            object_cad_asset_filepath = ObjectLibrary.get_cad_asset_path(
                object_library_folderpath=object_library.asset_folder_name,
                object_id=data['object_id'],
            ) 
            original_object_mesh = o3d.io.read_triangle_mesh(object_cad_asset_filepath)
            
            object_trajectory = []
            skip_flag = False
            for i, ns in enumerate(duration):
                
                headset_pose3d_with_dt = device_pose_provider.get_pose_at_timestamp(
                    timestamp_ns=ns,
                    time_query_options=option,
                    time_domain=time_domain,
                )
                object_poses_with_dt = (
                    object_pose_data_provider.get_pose_at_timestamp(
                        timestamp_ns=ns,
                        time_query_options=option,
                        time_domain=time_domain,
                    )
                )
                
                if headset_pose3d_with_dt is None or object_poses_with_dt is None:
                    skip_flag = True
                    break
                
                headset_pose3d = headset_pose3d_with_dt.pose3d
                T_world_device = headset_pose3d.T_world_device

                objects_pose3d_collection = object_poses_with_dt.pose3d_collection       
                
                if data['object_id'] in objects_pose3d_collection.poses.keys():
                    object_poses3d = objects_pose3d_collection.poses[data['object_id']]
                else:
                    # when the object is fully covered by hands
                    skip_flag = True
                    break
                
                T_world_object = object_poses3d.T_world_object
                
                if i == 0:
                    object_mesh = deepcopy(original_object_mesh)
                    center = object_mesh.get_center()
                    scale = 1e-3
                    object_mesh.scale(scale, center=center)
                    object_mesh.translate(-(center - center*scale))
                    object_bbox = object_mesh.get_axis_aligned_bounding_box()
                    bbox_vertices = np.asarray(object_bbox.get_box_points())
                    bbox_center = np.mean(bbox_vertices, axis=0)
                    
                world_device_coord = T_world_device.translation() # camera's translation in world coord. system
                world_device_pose = T_world_device.rotation().to_matrix() # camera's rotation in world coord. system
                world_object_coord = T_world_object.translation() # object's translation in world coord. system
                world_object_pose = T_world_object.rotation().to_matrix() # object's rotation in world coord. system
                # NOTE since object poses are provided based on the bottom center of object mesh, 
                # thus translate object coords to the center.
                diff = world_object_pose @ bbox_center 
                world_object_coord += diff
                
                # change object pose into the camera coord. system
                device_object_coord = (world_device_pose.T @ (world_object_coord - world_device_coord).T).T 
                device_object_pose = world_device_pose.T @ world_object_pose
                camera_object_coord = (R_intri.T @ (device_object_coord - T_intri).T).T
                camera_object_pose = R_intri.T @ device_object_pose
                camera_object_quat = R.from_matrix(camera_object_pose).as_quat()
                
                if i == 0:
                    obs_pil_image = Image.open(f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{round(ns, 3)}.jpg")
                    depth = depth_anything.get_only_depth(
                        obs_pil_image, 
                        camera_cfg.devices.aria.pinhole_image_size, 
                        camera_cfg.devices.aria.pinhole_image_size
                    )
                    image_coord = pinhole_cw90.project(camera_object_coord.T)
                    if image_coord is None:
                        # object's out of frame
                        skip_flag = True
                        break
                    else:
                        image_coord = image_coord.astype(int)
                    
                    # NOTE: since the scale of actual depth and qseudo depth is different,
                    # we align them to pseudo depths.
                    ratio_depth = depth[image_coord[1]][image_coord[0]] / camera_object_coord[0][-1]
                    
                    # mesh: mm scale, world: m scale
                    # thus, mesh * 1e-3 = world
                    object_mesh = deepcopy(original_object_mesh)
                    center = object_mesh.get_center()
                    scale = ratio_depth * 1e-3
                    object_mesh.scale(scale, center=center)
                    object_mesh.translate(-(center - center*scale))
                    object_bbox = object_mesh.get_axis_aligned_bounding_box()
                    bbox_vertices = np.asarray(object_bbox.get_box_points())
                
                camera_object_coord *= ratio_depth
                camera_object_6dof = np.concatenate([np.squeeze(camera_object_coord), camera_object_quat], axis=0)
                object_trajectory.append(camera_object_6dof)

            if skip_flag or not len(object_trajectory): 
                continue
            
            init_bbox_center = np.mean(bbox_vertices, axis=0)
            bbox_vertices -= init_bbox_center
            object_trajectory = np.stack(object_trajectory)
            
            traj = {
                "init_bbox": bbox_vertices,
                "traj_quat": object_trajectory
            }
            
            if args.visualize:
                object_trajectory[:,0] = camera_cfg.devices.aria.focal_len*object_trajectory[:,0]/object_trajectory[:,2] + camera_cfg.devices.aria.principal_point
                object_trajectory[:,1] = camera_cfg.devices.aria.focal_len*object_trajectory[:,1]/object_trajectory[:,2] + camera_cfg.devices.aria.principal_point
                
                plt.imshow(np.array(obs_pil_image))
                plt.plot(object_trajectory[:,0], object_trajectory[:,1], c='red')
                plt.savefig('temp.jpg')
                plt.clf()
                import pdb; pdb.set_trace()
            else:
                os.makedirs(f'{args.save_dir}/obs_images/{dataset_name}/{video_uid}', exist_ok=True)
                os.makedirs(f'{args.save_dir}/depths/{dataset_name}/{video_uid}', exist_ok=True)
                os.makedirs(f'{args.save_dir}/trajs/{dataset_name}/{video_uid}', exist_ok=True)
                obs_pil_image.save(f'{args.save_dir}/obs_images/{dataset_name}/{video_uid}/{file_name}.jpg')
                np.save(f'{args.save_dir}/depths/{dataset_name}/{video_uid}/{file_name}', depth)
                with open(f'{args.save_dir}/trajs/{dataset_name}/{video_uid}/{file_name}.pkl', 'wb') as f:
                    pickle.dump(traj, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_resource", 
        default='/home/yoshida/ULAT/data/Depth-Anything-V2/ckpts/depth_anything_v2_metric_hypersim_vitl.pth'
    )
    
    parser.add_argument('--root_dir', default="/data/g-liat/yoshida/Hot3D")
    parser.add_argument('--save_dir', default="/data/g-liat/yoshida/EgoScaler")
    parser.add_argument('--visualize', action="store_true")
    
    args = parser.parse_args()
    
    main(args)