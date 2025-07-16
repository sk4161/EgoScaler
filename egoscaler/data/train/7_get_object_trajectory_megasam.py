"""
This script extracts object 6DoF trajectory from a single egocentric video clip
"""

import os
import json
import pickle
import gc
import argparse
import psutil
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from egoscaler.configs import CameraConfig as camera_cfg, DataConfig as data_cfg
from egoscaler.data.tools import (
    iou,
    get_points_colors,
    prepare_dataset,
    execute_global_registration,
    refine_registration,
    compute_rotation,
    minimum_3Dbox,
)


def main(args):
    with open(f"{args.data_dir}/infos.json", "r") as f:
        all_data = json.load(f)

    # for multi processing
    print(f"{args.start_index} to {args.end_index}.")
    # Restrict index range if specified
    if not (args.start_index == 0 and args.end_index == -1):
        all_data = all_data[
            args.start_index : args.end_index if args.end_index != -1 else None
        ]

    for idx, data in enumerate(tqdm(all_data, desc="Processing Data")):
        if "dataset_name" in data:
            dataset_name = data["dataset_name"]
        else:
            dataset_name = "hot3d"
        video_uid = data["video_uid"]
        file_name = data.get("file_name", "")

        action_desc = data["action_description"]

        sampling_rate = 1 / camera_cfg.fps
        timestamp = data["timestamp"]
        start_sec = data["start_sec"]
        end_sec = data["end_sec"]

        original_duration = np.round(
            np.arange(
                timestamp - camera_cfg.time_window,
                timestamp + camera_cfg.time_window,
                sampling_rate,
            ),
            9,
        )
        npz_file_path = args.npz_file_path
        data = np.load(npz_file_path)
        npz_file_path_pc = args.pc_npz_file_path
        pc_data = np.load(npz_file_path_pc)
        depth = data["depths"]  # (N, H, W)
        images = data["images"]  # (N, H, W, 3)
        intrinsic = data["intrinsic"]  # (3, 3)
        cam_c2w = data["cam_c2w"]  # (N, 4, 4)

        point_maps = pc_data["point_maps"]  # (N, H, W, 3)
        color_maps = pc_data["color_maps"]  # (N, H, W, 3)

        width, height = images.shape[2], images.shape[1]

        print(f"Start sec: {start_sec}, End sec: {end_sec}")
        start_index = np.where(np.round(original_duration, 3) == round(start_sec, 3))[0]
        end_index = np.where(np.round(original_duration, 3) == round(end_sec, 3))[0]
        print(f"Start index: {start_index}, End index: {end_index}")

        # camera intrinsics
        focal_len_x = intrinsic[0, 0]
        focal_len_y = intrinsic[1, 1]
        principal_point_x = intrinsic[0, 2]
        principal_point_y = intrinsic[1, 2]

        # obs info
        pil_image_dir = f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}"
        if not os.path.exists(pil_image_dir):
            print(f"Image directory {pil_image_dir} does not exist. Skipping.")
            continue

        image_files = sorted(
            [
                f
                for f in os.listdir(pil_image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ],
            key=lambda x: int(os.path.splitext(x)[0]),
        )
        if not image_files:
            print(f"No images found in {pil_image_dir}. Skipping.")
            continue

        pil_image_path = os.path.join(pil_image_dir, image_files[start_index[0]])
        if not os.path.exists(pil_image_path):
            print(f"Image file {pil_image_path} does not exist. Skipping.")
            continue

        pil_image = Image.open(pil_image_path).convert("RGB")
        # resize to images size
        pil_image = pil_image.resize((width, height))
        image = np.array(pil_image)

        object_points_sequence = []
        object_colors_sequence = []
        for frame_idx in range(start_index[0], end_index[0] + 1):
            object_pc_path = f"{args.monst3r_output_dir}/object_pointclouds/frame_{frame_idx:05d}.npy"
            object_color_path = (
                f"{args.monst3r_output_dir}/object_colors/frame_{frame_idx:05d}.npy"
            )

            if not os.path.exists(object_pc_path) or not os.path.exists(
                object_color_path
            ):
                print(
                    f"Object point cloud files for index {frame_idx} do not exist. Skipping."
                )
                continue

            object_points = np.load(object_pc_path)
            object_colors = np.load(object_color_path)

            if object_points.shape[0] == 0:
                print(f"No points found in {object_pc_path}. Skipping.")
                continue

            object_points_sequence.append(object_points)
            object_colors_sequence.append(object_colors)

        ############################################################################################################

        # trajectory projection
        ############################################################################################################
        start = time.time()
        voxel_size = data_cfg.pcm_cfg.voxel_size * 0.1  # 0.01

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(object_points_sequence[0])
        target.colors = o3d.utility.Vector3dVector(object_colors_sequence[0])
        search_radius = voxel_size * 3.0
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=search_radius, max_nn=30
            )
        )

        transform_matrices = []
        transform_matrices.append(np.identity(4))
        projected_traj = []
        regist_flag = True

        absolute_rotation = np.eye(3)
        absolute_position = object_points_sequence[0].mean(axis=0)
        init_bbox = minimum_3Dbox(object_points_sequence[0])

        if init_bbox is None:  # Failed to create 3D bbox
            regist_flag = False
            break

        quaternion = R.from_matrix(absolute_rotation).as_quat()
        projected_traj.append(np.concatenate([absolute_position, quaternion]))

        for i in range(1, len(object_points_sequence)):
            current_object_point = object_points_sequence[i]
            current_object_color = object_colors_sequence[i]

            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(current_object_point)
            source.colors = o3d.utility.Vector3dVector(current_object_color)
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_radius, max_nn=30
                )
            )

            source_keypoints_current, source_feature_current = prepare_dataset(
                source, voxel_size
            )
            target_keypoints, target_feature = prepare_dataset(target, voxel_size)

            # --- DEBUG LOGGING ---
            print(
                f"Frame {i}: Source Keypoints: {source_keypoints_current.points.__len__()}"
            )
            print(f"Frame {i}: Target Keypoints: {target_keypoints.points.__len__()}")
            if source_feature_current is not None:
                print(
                    f"Frame {i}: Source Feature data shape: {source_feature_current.data.shape}"
                )
            # --- END DEBUG LOGGING ---

            # global_matching
            result_ransac = execute_global_registration(
                source_keypoints_current,
                target_keypoints,
                source_feature_current,
                target_feature,
                voxel_size,
            )

            # --- DEBUG LOGGING ---
            if result_ransac is not None:
                print(
                    f"Frame {i}: RANSAC Registration Fitness: {result_ransac.fitness:.4f}, Inlier RMSE: {result_ransac.inlier_rmse:.4f}"
                )
            else:
                print(f"Frame {i}: RANSAC Registration failed or returned None.")
            # --- END DEBUG LOGGING ---

            # --- DEBUG LOGGING ---
            print(
                f"Frame {i}: Source Point Cloud Size: {len(source.points)}, Target Point Cloud Size: {len(target.points)}"
            )
            # --- END DEBUG LOGGING ---

            # # local matching
            # result_icp = refine_registration(
            #     source,
            #     target,
            #     result_ransac,
            #     voxel_size,
            # )

            # # --- DEBUG LOGGING ---
            # if result_icp is not None:
            #     print(
            #         f"Frame {i}: ICP Registration Fitness: {result_icp.fitness:.4f}, Inlier RMSE: {result_icp.inlier_rmse:.4f}"
            #     )
            # else:
            #     print(f"Frame {i}: ICP Registration failed or returned None.")
            # # --- END DEBUG LOGGING ---

            if result_ransac is None:  # Point cloud registration failed.
                regist_flag = False
                break
            else:
                transform_matrices.append(result_ransac.transformation)

            target = source

            # projection to initial frame
            transform = transform_matrices[0]
            for j in range(1, i + 1):
                transform = np.dot(transform, transform_matrices[j])

            absolute_rotation = transform[:3, :3]
            local_center = current_object_point.mean(axis=0)
            homog = np.concatenate([local_center, [1]])
            absolute_position = (transform @ homog)[:3]

            quaternion = R.from_matrix(absolute_rotation).as_quat()
            projected_traj.append(np.concatenate([absolute_position, quaternion]))
        ############################################################################################################
        end = time.time()
        print(f"Time taken for registration: {end - start}", flush=True)

        if not regist_flag:
            continue

        traj_quat = np.stack(projected_traj)

        # project to first frame
        T_c2w_first = cam_c2w[start_index[0]]
        R_c2w_first = T_c2w_first[:3, :3]
        t_c2w_first = T_c2w_first[:3, 3]

        R_w2c_first = R_c2w_first.T
        t_w2c_first = -R_w2c_first @ t_c2w_first

        positions_world = traj_quat[:, 0:3]
        quats_world = traj_quat[:, 3:7]

        positions_aligned = (
            R_w2c_first @ positions_world.T + t_w2c_first[:, np.newaxis]
        ).T

        R_world_to_cam_obj = R.from_matrix(R_w2c_first)
        rotations_world_obj = R.from_quat(quats_world)

        rotations_aligned_obj = R_world_to_cam_obj * rotations_world_obj
        quats_aligned = rotations_aligned_obj.as_quat()

        traj_quat = np.hstack([positions_aligned, quats_aligned])

        positions = traj_quat[:, 0:3]
        quat = traj_quat[:, 3:7]

        rotation = R.from_quat(quat)
        rotvec = rotation.as_rotvec()

        traj_rotvec = np.hstack([positions, rotvec])

        init_bbox_center = np.mean(init_bbox, axis=0)
        init_bbox -= init_bbox_center

        traj = {
            "init_bbox": init_bbox,  # numpy:
            "traj_quat": traj_quat,  # numpy [n, 7],
            "traj_rotvec": traj_rotvec,  # numpy [n, 6]
        }

        if args.visualize:
            pil_image.save("./viz_data/image.jpg")
            # np.save("./viz_data/depth", obs_depth)
            with open("./viz_data/trajectory.pkl", "wb") as f:
                pickle.dump(traj, f)
            with open("./viz_data/text.txt", "w") as f:
                f.write(action_desc)

            traj_rotvec[:, 0] = (
                focal_len_x * traj_rotvec[:, 0] / traj_rotvec[:, 2] + principal_point_x
            )
            traj_rotvec[:, 1] = (
                focal_len_y * traj_rotvec[:, 1] / traj_rotvec[:, 2] + principal_point_y
            )

            plt.imshow(image)
            plt.plot(traj_rotvec[:, 0], traj_rotvec[:, 1], c="red")
            plt.savefig("./viz_data/traj.jpg")
            plt.clf()
            import pdb

            pdb.set_trace()
        else:
            os.makedirs(
                f"{args.save_dir}/megasam/obs_images/{dataset_name}/{video_uid}",
                exist_ok=True,
            )
            os.makedirs(
                f"{args.save_dir}/megasam/depths/{dataset_name}/{video_uid}",
                exist_ok=True,
            )
            os.makedirs(
                f"{args.save_dir}/megasam/trajs/{dataset_name}/{video_uid}",
                exist_ok=True,
            )

            pil_image.save(
                f"{args.save_dir}/megasam/obs_images/{dataset_name}/{video_uid}/{file_name}.jpg"
            )
            # np.save(
            #     f"{args.save_dir}/megasam/depths/{dataset_name}/{video_uid}/{file_name}",
            #     obs_depth,
            # )
            with open(
                f"{args.save_dir}/megasam/trajs/{dataset_name}/{video_uid}/{file_name}.pkl",
                "wb",
            ) as f:
                pickle.dump(traj, f)

        # release memory
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # depth anything
    parser.add_argument(
        "--pretrained_resource",
        default="/your/path/to/workdir/EgoScaler/egoscaler/data/third_party/Depth-Anything-V2/ckpts/depth_anything_v2_metric_hypersim_vitl.pth",
    )
    # SpaTracker
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/your/path/to/workdir/EgoScaler/egoscaler/data/third_party/SpaTracker/checkpoints/spaT_final.pth",
    )
    parser.add_argument("--model", type=str, default="cotracker", help="model name")
    parser.add_argument(
        "--downsample", type=float, default=0.8, help="downsample factor"
    )
    parser.add_argument("--grid_size", type=int, default=100, help="grid size")
    parser.add_argument(
        "--outdir", type=str, default="./vis_results", help="output directory"
    )
    parser.add_argument("--fps", type=float, default=1, help="fps")
    parser.add_argument("--len_track", type=int, default=10, help="len_track")
    parser.add_argument("--fps_vis", type=int, default=30, help="len_track")
    parser.add_argument("--crop", action="store_true", help="whether to crop the video")
    parser.add_argument(
        "--crop_factor", type=float, default=1, help="whether to crop the video"
    )
    parser.add_argument(
        "--backward", action="store_true", help="whether to backward the tracking"
    )
    parser.add_argument(
        "--vis_support",
        action="store_true",
        help="whether to visualize the support points",
    )
    parser.add_argument("--query_frame", type=int, default=0, help="query frame")
    parser.add_argument("--point_size", type=int, default=3, help="point size")
    parser.add_argument(
        "--rgbd", action="store_true", help="whether to take the RGBD as input"
    )

    # data dirs
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_dir", default="/your/path/to/savedir/EgoScaler")
    parser.add_argument("--visualize", action="store_true")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)

    # intermediate results
    parser.add_argument("--monst3r_output_dir", default="./demo_tmp")
    parser.add_argument(
        "--npz_file_path",
        default="/home/kanazawa/egovision/EgoScaler/egoscaler/data/third_party/mega-sam/outputs_cvd/hoge_sgd_cvd_hr.npz",
    )
    parser.add_argument(
        "--pc_npz_file_path",
        default="/home/kanazawa/egovision/EgoScaler/output_maps.npz",
    )

    args = parser.parse_args()

    main(args)
