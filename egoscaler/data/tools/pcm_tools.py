import open3d as o3d
import numpy as np
from scipy.linalg import svd

def preprocess_point_cloud(pcd, voxel_size):
    # Create a voxel grid filter to downsample the point cloud
    # treat the center of the voxel as a keypoint
    keypoints = pcd.voxel_down_sample(voxel_size)

    # estimate normals for the downsampled point cloud
    radius_normal = voxel_size * 2
    # compute normals for the keypoints using a hybrid KDTree search; max_nn is the maximum number of neighbors to consider
    keypoints.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # extracting FPFH features requires keypoints and normals
    # to compute FPFH features, we need to specify a radius for the feature computation
    # the radius should be larger than the voxel size to ensure that we capture enough local structure
    radius_feature = voxel_size * 5
    # use max_nn to limit the number of neighbors considered for feature computation
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return keypoints, feature
    
def prepare_dataset(pcd, voxel_size):
    # extract keypoints and features using voxel downsampling
    keypoints, feature = preprocess_point_cloud(pcd, voxel_size)
    return keypoints, feature

def execute_global_registration(source_keypoints, target_keypoints, source_feature, target_feature,  voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_keypoints,
        target_keypoints,
        source_feature,
        target_feature,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    try:
        result = o3d.pipelines.registration.registration_colored_icp(
        #result = o3d.t.pipelines.registration.multi_scale_icp(
            source,
            target,
            distance_threshold,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=30)
            #o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    except Exception as e:
        print(e)
        result = None
        
    return result

def get_points_colors(rgbd, bbox, width, height, principal_p, focal_len_x, focal_len_y, d_thres=None):
        
    image = rgbd[:,:,:3]
    z = rgbd[:,:,-1]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - principal_p) / focal_len_x
    y = (y - principal_p) / focal_len_y
    
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3) / 255.0
    valid_color_indices = np.all(image != 0, axis=2)
    mask = np.ones((height, width))
    if bbox is not None:
        for box in bbox:
            mask[box['box']['ymin']:box['box']['ymax'], box['box']['xmin']:box['box']['xmax']] = 0 
    mask = mask[..., np.newaxis]
    static_object_indices = np.all(mask != 0, axis=2)
    
    if d_thres is not None:
        valid_depth_indices = z < d_thres
        valid_indices = valid_depth_indices & valid_color_indices & static_object_indices
    else:
        valid_indices = valid_color_indices & static_object_indices
    
    points = points[valid_indices.ravel(), :] 
    colors = colors[valid_indices.ravel(), :]   
    
    return points, colors


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0  
    return intersection / union

def minimum_3Dbox(points):
    if len(points) < 5:
        print(f"Not enough points ({len(points)}) to compute oriented bounding box.")
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    clean_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    try:
        oriented_bounding_box = clean_pcd.get_oriented_bounding_box()
    except Exception as e:
        print(f"Error in computing oriented bounding box: {e}")
        return None

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