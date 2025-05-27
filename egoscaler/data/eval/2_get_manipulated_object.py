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
import numpy as np
import json
from PIL import Image
from collections import defaultdict
import argparse
from egoscaler.configs import CameraConfig as camera_cfg

def main(args):
    object_library_path = os.path.join(args.root_dir, "assets")
    object_library = load_object_library(object_library_folderpath=object_library_path)
    sequence_paths = glob(f'{args.root_dir}/P*')
    dataset_name = 'hot3d'
    
    counter = 0
    for sequence_path in tqdm(sequence_paths):
        try:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=sequence_path,
                object_library=object_library,
                mano_hand_model=None,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        
        # since Meta Quest not contain rgb data, just skipping
        if hot3d_data_provider.get_device_type() != Headset.Aria:
            continue
        
        video_uid = sequence_path.split('/')[-1]
        provider = data_provider.create_vrs_data_provider(f"{sequence_path}/recording.vrs")
        
        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)

        time_domain = TimeDomain.TIME_CODE
        option = TimeQueryOptions.CLOSEST
        timestamps: list = provider.get_timestamps_ns(stream_id, time_domain)
        video_start_ns, video_end_ns = timestamps[0], timestamps[-1]
        
        # split whole video into 4 sec segmments
        segments = np.arange(video_start_ns, video_end_ns, 2*camera_cfg.time_window*1e+9)
        
        for start_ns, end_ns in zip(segments, segments[1:]):
            sampling_rate = 1 / camera_cfg.fps
            duration = np.arange(start_ns, end_ns, 1e+9 * sampling_rate).astype(int)
            duration = duration[np.round(np.arange(0, len(duration), len(duration)//8))] # for faster processing
            
            object_trajectories = defaultdict(list)
            
            timestamp: float = (start_ns + end_ns) / (2 * 1e+9) # sec.
            file_name = str(int((start_ns + end_ns) / 2))
            
            for ns in duration:
                
                object_poses_with_dt = (
                    object_pose_data_provider.get_pose_at_timestamp(
                        timestamp_ns=ns,
                        time_query_options=option,
                        time_domain=time_domain,
                    )
                )
                
                if object_poses_with_dt is None:
                    continue

                objects_pose3d_collection = object_poses_with_dt.pose3d_collection
                
                for (
                    object_uid,
                    object_pose3d,
                ) in objects_pose3d_collection.poses.items():
                    object_name = object_library.object_id_to_name_dict[object_uid]
                    object_name = object_name + "|" + str(object_uid)
                    T_world_object = object_pose3d.T_world_object
                    world_object_coord = np.squeeze(T_world_object.translation())
                    
                    object_trajectories[object_name].append(world_object_coord)
                    
            object_trajectories = {_key: np.stack(_val) for _key, _val in object_trajectories.items()}   
            
            diff = []
            object_names = []
            for object_name, object_trajectory in object_trajectories.items():
                trajectory = object_trajectory.copy()

                deltas = np.diff(trajectory, axis=0)
                distances = np.linalg.norm(deltas, axis=1)
                total_change = np.sum(distances)
                
                diff.append(total_change)
                object_names.append(object_name)
             
            diff = np.stack(diff)
            
            if np.all(diff<=1e-1):
                # no object moves over 10cm
                continue
            else:
                manipulated_object_index = np.argmax(diff)
                manipulated_object = object_names[manipulated_object_index]
            
            info = {
                "video_uid": video_uid,
                "object_id": manipulated_object.split('|')[1],
                "manipulated_object": manipulated_object.split('|')[0],
                "timestamp": timestamp,
                "file_name": file_name,
            }
            
            os.makedirs(f'{args.save_dir}/infos/{dataset_name}/{video_uid}', exist_ok=True)
            with open(f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json', 'w') as f:
                json.dump(info, f)
            
            counter += 1
        
    print(f"Created info: {counter}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_dir', default="/data/g-liat/yoshida/Hot3D")
    parser.add_argument('--save_dir', default="/data/g-liat/yoshida/EgoScaler")
    
    args = parser.parse_args()
    
    main(args)