import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd
import open3d as o3d
import os

def judge(point, ray1, ray2, ray3, ray4, translation):
    x, y, z = point
    x0, y0, z0 = translation
    val1 = ray1[0]*(x-x0) + ray1[1]*(y-y0) + ray1[2]*(z-z0)
    val2 = ray2[0]*(x-x0) + ray2[1]*(y-y0) + ray2[2]*(z-z0)
    val3 = ray3[0]*(x-x0) + ray3[1]*(y-y0) + ray3[2]*(z-z0)
    val4 = ray4[0]*(x-x0) + ray4[1]*(y-y0) + ray4[2]*(z-z0)
    if val1 < 0 and val2 > 0 and val3 > 0 and val4 < 0:
        return True
    else:
        return False

def mask_from_hod(hod_res, height, width):
    hand_mask = np.ones((height, width))
    obj_mask = np.ones((height, width))
    hand_bbox = hod_res['hand-bbox']
    obj_bbox = hod_res['obj-bbox']
    
    for h_bbox in hand_bbox:
        h_bbox = h_bbox[:4]
        hand_mask[h_bbox[1]:h_bbox[3], h_bbox[0]:h_bbox[2]] = 0
        
    for o_bbox in obj_bbox:
        obj_mask[o_bbox[1]:o_bbox[3], o_bbox[0]:o_bbox[2]] = 0
              
    return hand_mask * obj_mask

def get_normal_vec(vec1, vec2):
    return np.cross(vec1, vec2)

def cropped_point_cloud(points, image, T_world_from_device, T_device_from_camera, camera_calib):
    edge1 = [0, 0] # pixel_coords
    edge2 = [0, image.shape[1]]
    edge3 = [image.shape[0], 0]
    edge4 = [image.shape[0], image.shape[1]]
    
    device_ray1 = T_device_from_camera @ camera_calib.unproject_no_checks(edge1)
    device_ray2 = T_device_from_camera @ camera_calib.unproject_no_checks(edge2) 
    device_ray3 = T_device_from_camera @ camera_calib.unproject_no_checks(edge3) 
    device_ray4 = T_device_from_camera @ camera_calib.unproject_no_checks(edge4) 
    
    normal_vec1 = np.dot(T_world_from_device.rotation().to_matrix(), get_normal_vec(device_ray1[:,0], device_ray2[:,0])) #法線vec
    normal_vec2 = np.dot(T_world_from_device.rotation().to_matrix(), get_normal_vec(device_ray1[:,0], device_ray3[:,0]))
    normal_vec3 = np.dot(T_world_from_device.rotation().to_matrix(), get_normal_vec(device_ray3[:,0], device_ray4[:,0]))
    normal_vec4 = np.dot(T_world_from_device.rotation().to_matrix(), get_normal_vec(device_ray2[:,0], device_ray4[:,0]))
    
    view_points = []
    for point in tqdm(points):
        if judge(point.position_world, normal_vec1, normal_vec2, normal_vec3, normal_vec4,
                T_world_from_device.translation()[0]):
            view_points.append([_ for _ in point.position_world])
            
    return view_points

def multiply_homo(homographies, t1, t2):
    # this func returns t2->t1 homo
    # -> t1 coords = (this func' output) * t2 coords
    global_h = None
    if t1 == t2:
        return np.eye(3)

    for t in homographies:
        
        h = homographies[t]
        
        t = float(t)
        if h is not None:
            h = np.array(h)
    
        if t == t1:
            if h is None:
                return None
            else:
                global_h = h
        
        elif t1 < t < t2:
            if h is None:
                continue
            else:
                if global_h is None:
                    global_h = h
                else:
                    global_h = np.dot(global_h, h)

        elif t >= t2:
            if h is not None and global_h is None:
                global_h = h
            break
    
    return global_h

def depth_alignment(image, obs_depth, depth, obs_mask, mask, homo):
    bin_image = image.sum(axis=2).astype(bool)
    
    depth *= bin_image
    obs_depth *= bin_image
    
    depth = cv2.warpPerspective(depth, homo, (1408, 1408))
    mask = cv2.warpPerspective(mask.astype(float), homo, (1408, 1408))
    
    common_mask = obs_mask * mask * bin_image
    
    active_obs_depth = obs_depth[common_mask.nonzero()]
    active_depth = depth[common_mask.nonzero()]  
    
    diff = (active_obs_depth - active_depth)
    diff = np.where(abs(diff) > 1.5, 0, diff).mean()
    #diff = diff[diff.nonzero()].mean()
    
    return diff

def active_hand(obj_masks, obj_region):
    # get the nearest hand to the center of the object
    obj_mean_y, obj_mean_x = np.argwhere(obj_region).mean(axis=0)
    obj_masks = obj_masks.numpy()
    
    distance = []
    for obj_mask in obj_masks:
        obj_mask = obj_mask
        mean_y, mean_x = np.argwhere(obj_mask).mean(axis=0)    
        dis = np.sqrt((mean_x - obj_mean_x)**2 + (mean_y - obj_mean_y)**2)
        distance.append(dis)

    distance = np.stack(distance)
    indx = np.argmin(distance)
    
    return obj_masks[indx]

def get_mask_from_narr(masks, left_or_right, width, height):
    
    if not len(masks):
        return None
    
    masks = masks.astype(np.float32)
    
    centers = []
    for mask in masks:
        true_points = np.where(mask)
        centers.append([true_points[0].mean(), true_points[1].mean()])
    
    centers = np.stack(centers)
    
    if left_or_right == 'right':
        pivot_h, pivot_v = width, height
    elif left_or_right == 'left':
        pivot_h, pivot_v = 0, height
    
    weights_h = ((pivot_h - centers[:,0])**2)
    weights_v = ((pivot_v - centers[:,1])**2)
    
    weights = weights_v + weights_h
    
    index = np.argmin(weights)
    
    return masks[index]

def visualize_traj(track1, track2, img):
    """
    track1: last points [y, x]
    track2: behind track1
    img: PIL
    """
    
def get_nearest_tool_mask(tool_masks, object_mask):
    """
    tool_masks, object_mask: binary numpy 
    """
    if not tool_masks.shape[0]:
        return None, None
    
    _xs, _ys = np.where(object_mask)
    _x, _y = _xs.mean(), _ys.mean()
    
    nearest_index = 0
    distance = 1e+4
    for index, tool_mask in enumerate(tool_masks):
        xs, ys = np.where(tool_mask)
        x, y = xs.mean(), ys.mean()
        dist = np.sqrt((x-_x)**2 + (y-_y)**2)
        if dist < distance:
            distance = dist
            nearest_index = index
            
    # get closest point of tool mask for object_mask
    xs, ys = np.where(tool_masks[nearest_index])
    
    nearest_point_index = np.argmin(np.sqrt((xs - _x)**2 + (ys - _y)**2))
    init_coords = np.array([xs[nearest_point_index], ys[nearest_point_index]])
    return tool_masks[nearest_index].astype(np.float32), init_coords

def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0  
    return intersection / union

def minimum_3Dbox(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        oriented_bounding_box = pcd.get_oriented_bounding_box()
    except Exception as e:
        print(e)
        return None
    #bounding_box = pcd.get_axis_aligned_bounding_box()
    obb_vertices = np.asarray(oriented_bounding_box.get_box_points())
    return obb_vertices

def compute_rotation(initial_points, final_points):
    """
    Compute the rotation matrix that aligns initial_points to final_points.
    
    Parameters:
    initial_points (np.ndarray): N x 3 array of initial 3D points
    final_points (np.ndarray): N x 3 array of final 3D points

    Returns:
    R (np.ndarray): 3 x 3 rotation matrix
    """
    # Compute centroids
    centroid_initial = np.mean(initial_points, axis=0)
    centroid_final = np.mean(final_points, axis=0)
    
    # Center the points
    initial_centered = initial_points - centroid_initial
    final_centered = final_points - centroid_final
    
    # Compute the covariance matrix
    H = initial_centered.T @ final_centered
    
    # Perform SVD
    U, S, Vt = svd(H)
    V = Vt.T
    
    # Compute rotation matrix
    R = V @ U.T
    
    # Ensure a proper rotation (det(R) should be 1)
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T
    
    return R

def is_image_valid(image_file):
    """
    Check if the image file is valid and not corrupted.
    
    :param image_file: path to the image file
    :return: True if the image is valid, False otherwise
    """
    if not os.path.exists(image_file):
        return False
    try:
        with Image.open(image_file) as img:
            img.verify()  # Verify that the image is not corrupted
        return True
    except (IOError, UnidentifiedImageError):
        return False