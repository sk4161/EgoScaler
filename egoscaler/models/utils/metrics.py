import re
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation as R

def final_displacement_error(gen_traj: np.array, gt_traj: np.array) -> float:
    
    len_gen = gen_traj.shape[0]
    len_gt = gt_traj.shape[0]
    
    if len_gen > len_gt:
        gen_traj_padded = gen_traj[:len_gt, :]
    elif len_gen < len_gt:
        pad_size = len_gt - len_gen
        last_point = gen_traj[-1, :].reshape(1, -1)
        padding = np.repeat(last_point, pad_size, axis=0)
        gen_traj_padded = np.vstack([gen_traj, padding])
    else:
        gen_traj_padded = gen_traj    

    final_gen = gen_traj_padded[-1]
    final_gt = gt_traj[-1]
    
    diff = np.linalg.norm(final_gt - final_gen, ord=2)
    
    return diff

def initial_displacement_error(gen_traj: np.array, gt_traj: np.array) -> float:
    
    init_gen = gen_traj[0]
    init_gt = gt_traj[0]
    
    diff = np.linalg.norm(init_gt - init_gen, ord=2)
    
    return diff

def average_displacement_error(gen_traj: np.ndarray, gt_traj: np.ndarray) -> float:

    len_gen = gen_traj.shape[0]
    len_gt = gt_traj.shape[0]
    
    if len_gen > len_gt:
        gen_traj_padded = gen_traj[:len_gt, :]
    elif len_gen < len_gt:
        pad_size = len_gt - len_gen
        last_point = gen_traj[-1, :].reshape(1, -1)
        padding = np.repeat(last_point, pad_size, axis=0)
        gen_traj_padded = np.vstack([gen_traj, padding])
    else:
        gen_traj_padded = gen_traj
    
    diff = np.linalg.norm(gt_traj - gen_traj_padded, ord=2, axis=1).mean()
    
    return diff

def dynamic_time_warping(gen_traj: np.array, gt_traj: np.array) -> float:
    distance, path = fastdtw(gen_traj, gt_traj, dist=euclidean)
    return distance

def anglar_distance(gen_rot: np.ndarray, gt_rot: np.ndarray) -> float:
    len_gen = gen_rot.shape[0]
    len_gt = gt_rot.shape[0]
    
    if len_gen > len_gt:
        gen_rot_padded = gen_rot[:len_gt, :]
    elif len_gen < len_gt:
        pad_size = len_gt - len_gen
        last_rot = gen_rot[-1, :].reshape(1, -1)
        padding = np.repeat(last_rot, pad_size, axis=0)
        gen_rot_padded = np.vstack([gen_rot, padding])
    else:
        gen_rot_padded = gen_rot
        
    assert gen_rot_padded.shape[0] == gt_rot.shape[0]
    
    ad = []
    for gen_r, gt_r in zip(gen_rot_padded, gt_rot):
        gen_quat = R.from_rotvec(gen_r).as_quat()
        gt_quat = R.from_rotvec(gt_r).as_quat()
        
        dot_product = np.dot(gen_quat, gt_quat)
        
        angle_dist = 2 * np.arccos(np.clip(dot_product, -1.0, 1.0))
        ad.append(angle_dist)
    
    # average angular distance
    return sum(ad) / len(ad)