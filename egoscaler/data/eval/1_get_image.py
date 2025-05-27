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
import argparse
import re
from PIL import Image, ImageDraw, ImageFont
from egoscaler.configs import CameraConfig as camera_cfg

def main(args):    
    object_library_path = os.path.join(args.root_dir, "assets")
    object_library = load_object_library(object_library_folderpath=object_library_path)
    sequence_paths = glob(f'{args.root_dir}/P*')
    dataset_name = 'hot3d'

    for sequence_path in tqdm(sequence_paths):
        try:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=sequence_path,
                object_library=object_library,
                mano_hand_model=None,
            )
        except Exception as e:
            print(e)
            continue
        
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
        time_domain = TimeDomain.TIME_CODE
        option = TimeQueryOptions.CLOSEST
        timestamps: list = provider.get_timestamps_ns(stream_id, time_domain)
        video_start_ns, video_end_ns = timestamps[0], timestamps[-1]
        
        # split whole video into 4 sec segmments
        segments = np.arange(video_start_ns, video_end_ns, 2*camera_cfg.time_window*1e+9)
        
        for start_ns, end_ns in zip(segments, segments[1:]):
            
            file_name = str(int((end_ns+start_ns) / 2))
            
            sampling_rate = 1 / camera_cfg.fps
            duration = np.arange(start_ns, end_ns, 1e+9 * sampling_rate).astype(int)
            
            if os.path.exists(f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}') and \
                len(glob(f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/*')) == len(duration):
                continue

            os.makedirs(f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}')
            for ns in duration:
                image = provider.get_image_data_by_time_ns(stream_id, ns, time_domain, option)[0].to_numpy_array()
                undis_image = calibration.distort_by_calibration(image, pinhole, rgb_camera_calibration)
                undis_image_cw90 = np.rot90(undis_image, k=3)
                pil_image = Image.fromarray(undis_image_cw90)
                # TODO: save as ns -> sec
                pil_image.save(f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{round(ns, 3)}.jpg')
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data dirs
    parser.add_argument("--root_dir", default="/data/g-liat/yoshida/Hot3D")
    parser.add_argument("--save_dir", default='/data/g-liat/yoshida/EgoScaler')
    
    args = parser.parse_args()
    
    main(args)