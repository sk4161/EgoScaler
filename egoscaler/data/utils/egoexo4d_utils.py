import json
from egoscaler.data.tools import hand_transfer_flag, process_hand_mentions
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from PIL import Image, UnidentifiedImageError
import numpy as np
from glob import glob
from egoscaler.configs import CameraConfig as camera_cfg
from egoscaler.data.tools import get_image, is_image_valid
import os


def load_annotations(split, args):
    with open(f'{args.root_egoexo4d_dir}/annotations/atomic_descriptions_{split}.json', 'r') as f:
        descriptions = json.load(f)['annotations']
    with open(f'{args.root_egoexo4d_dir}/takes.json', 'r') as f:
        takes = json.load(f)
    return descriptions, takes

def process_take(take, descriptions):
    video_uid = take['take_uid']
    task_name = take['parent_task_name']
    desc_infos = descriptions.get(video_uid, [{}])[0].get('descriptions', [])
    return video_uid, task_name, desc_infos

def process_description(desc_info):
    raw_desc = desc_info['text']
    not_interaction = hand_transfer_flag(raw_desc)
    raw_desc = process_hand_mentions(raw_desc)    
    return raw_desc, desc_info['timestamp'], desc_info['subject'], desc_info['ego_visible'], desc_info['unsure'], not_interaction

def extract_images(provider, data, save_path, video_duration=None):
    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)
    device_calibration = provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(camera_label)
    T_device_from_camera = rgb_camera_calibration.get_transform_device_camera()
    pinhole = calibration.get_linear_camera_calibration(
        camera_cfg.devices.aria.pinhole_image_size, 
        camera_cfg.devices.aria.pinhole_image_size, 
        camera_cfg.devices.aria.focal_len, 
        camera_label, T_device_from_camera
    )
    time_domain = TimeDomain.DEVICE_TIME
    option = TimeQueryOptions.CLOSEST
    start_ns = provider.get_first_time_ns(stream_id, time_domain)

    timestamp = data['timestamp']
    start_sec = timestamp - camera_cfg.time_window
    end_sec = timestamp + camera_cfg.time_window
    sampling_rate = 1 / camera_cfg.fps
    duration = np.arange(start_sec, end_sec, sampling_rate)

    if os.path.exists(save_path) and len(glob(f'{save_path}/*')) == len(duration):
        for _t in duration:
            image_file = os.path.join(save_path, f'{round(_t, 3)}.jpg')
            if is_image_valid(image_file):
                continue
            else:
                print(f"Existing image {image_file} is corrupted. Reprocessing.")
                os.remove(image_file) 
            image_array = get_image(
                _t, provider, start_ns, stream_id,
                time_domain, option, pinhole, rgb_camera_calibration
            )
            image = Image.fromarray(image_array)
            image.save(image_file)
        return
    
    os.makedirs(save_path, exist_ok=True)

    for _t in duration:
        image_array = get_image(
            _t, provider, start_ns, stream_id,
            time_domain, option, pinhole, rgb_camera_calibration
        )
        image = Image.fromarray(image_array)
        image_file = os.path.join(save_path, f'{round(_t, 3)}.jpg')
        image.save(image_file)