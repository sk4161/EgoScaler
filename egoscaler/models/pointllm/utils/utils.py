import re
import random
import numpy as np
import torch

from egoscaler.configs.camera import CameraConfig as camera_cfg

PINHOLE_IMAGE_HEIGHT = camera_cfg.devices.aria.pinhole_image_size
PINHOLE_IMAGE_WIDTH = camera_cfg.devices.aria.pinhole_image_size
FOCAL_LEN = camera_cfg.devices.aria.focal_length
PRICIPAL_POINT = camera_cfg.devices.aria.principal_point

def discretize_action(action_vector, num_bins=256):
    bins = np.linspace(-1, 1, num_bins)
    discrete_action = np.digitize(action_vector, bins) - 1
    return discrete_action.tolist()

def token_to_action(tokens, num_bins=256):
    bins = np.linspace(-1, 1, num_bins)
    continuous_action = [bins[val] for val in tokens]
    return continuous_action

def rt2_scaler(traj: np.array, maxmin: list, split: str) -> np.array:
    d_max, d_min = maxmin
    traj[:, [3,4,5]] = np.pi * traj[:, [3,4,5]]
    traj[:,2] = (1.0/2) * traj[:,2] + (1.0/2)
    traj[:,2] = (d_max - d_min) * traj[:,2] + d_min 
    
    traj[:,0] = (PINHOLE_IMAGE_WIDTH/2) * traj[:,0] + (PINHOLE_IMAGE_WIDTH/2)
    traj[:,0] = (traj[:,0] - PRICIPAL_POINT) * traj[:,2] / FOCAL_LEN
    traj[:,1] = (PINHOLE_IMAGE_HEIGHT/2) * traj[:,1] + (PINHOLE_IMAGE_HEIGHT/2)
    traj[:,1] = (traj[:,1] - PRICIPAL_POINT) * traj[:,2] / FOCAL_LEN

    return traj
    
def simple_scaler(traj: np.array, maxmin: list) -> np.array:
    
    d_max, d_min = maxmin
    traj[:, [3,4,5]] = np.pi * (2 * (traj[:, [3,4,5]] / 100) - 1)
    traj[:,2] = traj[:,2] / 100
    traj[:,2] = traj[:,2] * (d_max - d_min) + d_min
    traj[:,0] = (traj[:,0] - PRICIPAL_POINT) * traj[:,2] / FOCAL_LEN
    traj[:,1] = (traj[:,1] - PRICIPAL_POINT) * traj[:,2] / FOCAL_LEN
    
    return traj

def str_to_float(str, maxmin, split, rt2=False, only_pos=False, only_xy=False, z_values=None, num_bins=256):
    # rt2: the str is rt2 format or not.
    if rt2:
        if only_pos:
            pattern = re.compile(r'<p(\d+)> <p(\d+)> <p(\d+)>')
        elif only_xy:
            pattern = re.compile(r'<p(\d+)> <p(\d+)>')
        else:
            pattern = re.compile(r'<p(\d+)> <p(\d+)> <p(\d+)> <p(\d+)> <p(\d+)> <p(\d+)>')
    else:
        if only_pos:
            pattern = re.compile(r'<x(\d+)><y(\d+)><z(\d+)>')
        else:
            pattern = re.compile(r'<x(\d+)><y(\d+)><z(\d+)><rx(\d+)><ry(\d+)><rz(\d+)>')

    segments = str.split("<tsep>")
    
    traj = []
    last_traj = None 
    for i, seg in enumerate(segments):
        match = pattern.search(seg)
        if match:
            if rt2:
                if only_pos:
                    x, y, z = map(int, match.groups())
                    rx, ry, rz = 0, 0, 0
                    x, y, z, rx, ry, rz = token_to_action([x, y, z, rx, ry, rz], num_bins=num_bins)
                elif only_xy:
                    x, y = map(int, match.groups())
                    z, rx, ry, rz = 0, 0, 0, 0
                    x, y, z, rx, ry, rz = token_to_action([x, y, z, rx, ry, rz], num_bins=num_bins)
                    z = z_values[i] if i < len(z_values) else z_values[-1]
                else:
                    x, y, z, rx, ry, rz = map(int, match.groups())
                    x, y, z, rx, ry, rz = token_to_action([x, y, z, rx, ry, rz], num_bins=num_bins)
            else:
                if only_pos:
                    x, y, z = map(int, match.groups())
                    rx, ry, rz = 0, 0, 0
                else: 
                    x, y, z, rx, ry, rz = map(float, match.groups())
            current_traj = (x, y, z, rx, ry, rz)
            traj.append(current_traj)
            last_traj = current_traj
        else:
            if last_traj is not None:
                traj.append(last_traj)  # 前の値をコピー

    if len(traj):
        traj = np.array(traj).astype(np.float32)
        if rt2:
            traj = rt2_scaler(traj, maxmin, split)
        else:
            traj = simple_scaler(traj, maxmin)
    else:
        traj = None
        
    return traj   
