from projectaria_tools.core import calibration
from projectaria_tools.core import image as aria_image
import numpy as np

def convert_to_ns(start_ns, seconds):
    # start_ns is needed due to the annotated timestamp scale 
    # differ from aria recording scale
    return int(start_ns + seconds * 1e+9)

def get_image(sec, provider, start_ns, stream_id, time_domain, option, pinhole, cam_calibration):
    timestamp_ns = convert_to_ns(start_ns, sec)
    image = provider.get_image_data_by_time_ns(stream_id, timestamp_ns, time_domain, option)[0].to_numpy_array()
    undis_image = calibration.distort_by_calibration(image, pinhole, cam_calibration)
    undis_image_cw90 = np.rot90(undis_image, k=3)
    return undis_image_cw90

def get_key_timestamps(timestamp, narr_infos, is_prev_action):
    """
    is_prev_action: actions before interaction is are more important than after
    """
    sorted_timestamps = sorted([_['timestamp'] for _ in narr_infos])
    prev_timestamp = None
    next_timestamp = None
    for ts in sorted_timestamps:
        if ts < timestamp:
            prev_timestamp = ts
        elif ts > timestamp and next_timestamp is None:
            next_timestamp = ts
            break
    
    if is_prev_action:
        # get obs sec
        if prev_timestamp is None:
            obs_sec = timestamp - 0.5
        else:
            if timestamp - 0.5 < prev_timestamp:
                # interfering previous action
                obs_sec = (prev_timestamp + timestamp) / 2
            else:
                obs_sec = timestamp - 0.5
                
        # get inter sec
        if next_timestamp is None:
            inter_sec, inter_len = obs_sec, 1.0
        else:
            if timestamp + 1.0 > next_timestamp:
                inter_sec, inter_len = obs_sec, (timestamp + next_timestamp) / 2 - timestamp
            else:
                inter_sec, inter_len = obs_sec, 1.0
    
    else:
        # get obs sec
        if prev_timestamp is None:
            obs_sec = timestamp - 0.5
        else:
            if timestamp - 0.5 < prev_timestamp:
                # interfering previous action
                obs_sec = (prev_timestamp + timestamp) / 2
            else:
                obs_sec = timestamp - 0.5
                
        # get inter sec
        if next_timestamp is None:
            inter_sec, inter_len = timestamp, 1.0
        else:
            if timestamp + 1.0 > next_timestamp:
                inter_sec, inter_len = timestamp, (timestamp + next_timestamp) / 2 - timestamp + 0.2
            else:
                inter_sec, inter_len = timestamp, 1.0
    
    return obs_sec, inter_sec, inter_len