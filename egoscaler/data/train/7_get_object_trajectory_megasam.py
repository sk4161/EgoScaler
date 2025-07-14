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

from spatracker.predictor import SpaTrackerPredictor
from spatracker.utils.visualizer import Visualizer
from depth import DepthAnything
from egoscaler.data.train.tools.grounded_sam import GroundedSAM
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


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**2)  # memory usage in MB
    print(f"Memory Usage: {mem:.2f} MB")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load models
    depth_anything = DepthAnything(args, device)
    gsam_model = GroundedSAM(
        detector_id="IDEA-Research/grounding-dino-base",
        segmenter_id="facebook/sam-vit-huge",
        device=device,
    )
    spatrack = SpaTrackerPredictor(
        checkpoint=args.checkpoint, interp_shape=(384, 512), seq_length=12
    )
    spatrack.to(device)

    # memory usage log
    log_memory_usage()

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
        take_name = data.get("take_name", "")
        file_name = data.get("file_name", "")

        # Skip if trajectory already saved or metadata is invalid
        if os.path.exists(
            f"{args.save_dir}/trajs/{dataset_name}/{video_uid}/{file_name}.pkl"
        ):
            continue

        action_desc = data["action_description"]
        manipulated_object = data["manipulated_object"]

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
            3,
        )
        start_index = np.where(original_duration == round(start_sec, 3))[0]
        end_index = np.where(original_duration == round(end_sec, 3))[0]
        duration = original_duration[start_index[0] : end_index[0] + 1]

        # camera intrinsics
        focal_len_x = camera_cfg.devices.aria.focal_len
        focal_len_y = camera_cfg.devices.aria.focal_len
        principal_point = camera_cfg.devices.aria.principal_point

        # obs info
        start_sec_base = str(start_sec).replace(".", "")
        pil_image = Image.open(
            f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{start_sec_base}.jpg"
        )
        image = np.array(pil_image)
        width, height = pil_image.size
        obs_depth, obs_points, obs_colors = depth_anything.get_depth(
            pil_image=pil_image,
            final_width=width,
            final_height=height,
            focal_len_x=focal_len_x,
            focal_len_y=focal_len_y,
            principal_point=principal_point,
        )

        # get tracking results
        ############################################################################################################
        clip = []
        depths = []
        for i, _t in enumerate(duration):
            pil_img = Image.open(
                f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{_t}.jpg"
            )
            width, height = pil_img.size
            img = np.array(pil_img)
            clip.append(img)
            depth = depth_anything.get_only_depth(
                pil_image=pil_img, final_width=width, final_height=height
            )
            depths.append(depth)

        if not len(clip):
            continue

        # get target object mask
        object_masks, _, object_scores = gsam_model.predict(
            pil_image, [manipulated_object], threshold=data_cfg.mani_obj_det_threshold
        )
        if object_scores is None:  # target object not found
            continue

        # NOTE: hods filtering; not sure how much this is effective
        if os.path.exists(
            f"{args.save_dir}/hods/{dataset_name}/{video_uid}/{file_name}.pkl"
        ):
            with open(
                f"{args.save_dir}/hods/{dataset_name}/{video_uid}/{file_name}.pkl", "rb"
            ) as f:
                hod_results = pickle.load(f)
            hod_res = hod_results[start_sec]
        else:
            continue

        if len(hod_res["obj-bbox"]):
            hod_obj_mask = np.zeros_like(object_masks[0])
            hod_obj_mask[
                hod_res["obj-bbox"][0][1] : hod_res["obj-bbox"][0][3],
                hod_res["obj-bbox"][0][0] : hod_res["obj-bbox"][0][2],
            ] = 1
            ious = [iou(hod_obj_mask, obj_mask) for obj_mask in object_masks]
            target_obj_mask = object_masks[np.argmax(ious)]
        else:
            target_obj_mask = object_masks[
                np.argmax(object_scores)
            ]  # most confident object

        # tracking
        clip = np.stack(clip)
        depths = np.stack(depths)
        rgbd_seq = np.concatenate([clip, depths[:, :, :, np.newaxis]], axis=-1)

        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float()
        clip = clip.to(device)
        depths = torch.from_numpy(depths).float().to(device)[:, None]

        pred_tracks, pred_visibility, T_Firsts = spatrack(
            clip,
            video_depth=depths,
            grid_size=args.grid_size,
            segm_mask=torch.from_numpy(target_obj_mask)[None, None],
            wind_length=12,
        )
        msk_query = T_Firsts == args.query_frame
        pred_tracks = pred_tracks[:, :, msk_query.squeeze()]
        pred_visibility = pred_visibility[:, :, msk_query.squeeze()]
        pred_tracks = pred_tracks.cpu().detach()

        if args.visualize:
            vis = Visualizer(
                save_dir="./viz_data",
                pad_value=0,
                linewidth=3,
                show_first_frame=0,
                tracks_leave_trace=10,
                fps=10,
            )
            vis.visualize(
                clip,
                pred_tracks[..., :2],
                pred_visibility,
                query_frame=args.query_frame,
            )
        ############################################################################################################

        # trajectory projection
        ############################################################################################################
        if os.path.exists(
            f"{args.save_dir}/bboxes/{dataset_name}/{video_uid}/{file_name}.json"
        ):
            with open(
                f"{args.save_dir}/bboxes/{dataset_name}/{video_uid}/{file_name}.json",
                "r",
            ) as f:
                bboxes = json.load(f)
        else:
            continue

        d_thres = data_cfg.depth_threshold
        pred_tracks = pred_tracks.squeeze(0).numpy()
        depths = depths.squeeze(1).detach().cpu().numpy()
        points, colors = get_points_colors(
            rgbd=rgbd_seq[0],
            bbox=bboxes[str(start_sec)],
            width=width,
            height=height,
            principal_p=principal_point,
            focal_len_x=focal_len_x,
            focal_len_y=focal_len_y,
            d_thres=d_thres,
        )
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(points)
        target.colors = o3d.utility.Vector3dVector(colors)

        # TODO pred_track filtering
        xs = np.round(pred_tracks[:, :, 0]).astype(int)
        ys = np.round(pred_tracks[:, :, 1]).astype(int)
        validness = (
            (0 <= xs) & (xs < width) & (0 <= ys) & (ys < height)
        )  # check out of frame exists
        valid_frames = (
            np.sum(validness, axis=1) >= np.sum(validness[0]) / 2
        )  # if some of frames our of range beyound thresh, skip the instance

        if not np.all(valid_frames):
            continue

        valid_indices = np.all(validness, axis=0)

        transform_matrices = {}
        projected_traj = []
        regist_flag = True
        start = time.time()
        for i, (_t, coords, depth, rgbd) in enumerate(
            zip(duration, pred_tracks, depths, rgbd_seq)
        ):
            xs, ys, zs = (
                np.round(coords[:, 0]).astype(int),
                np.round(coords[:, 1]).astype(int),
                coords[:, 2],
            )
            xs, ys, zs = xs[valid_indices], ys[valid_indices], zs[valid_indices]

            ratio_depth = np.mean(depth[ys, xs] / zs)

            xs = (xs - principal_point) / focal_len_x
            ys = (ys - principal_point) / focal_len_y
            xs *= zs
            ys *= zs
            object_coords = np.array([xs, ys, zs]).T

            if i == 0:
                absolute_rotation = np.eye(3)
                absolute_position = object_coords.mean(axis=0)
                init_bbox = minimum_3Dbox(np.array([xs, ys, zs]).T)
                init_coords = object_coords.copy()
                init_rotation = absolute_rotation.copy()

                if init_bbox is None:  # Failed to create 3D bbox
                    regist_flag = False
                    break

            else:
                points, colors = get_points_colors(
                    rgbd=rgbd,
                    bbox=bboxes[str(_t)],
                    width=width,
                    height=height,
                    principal_p=principal_point,
                    focal_len_x=focal_len_x,
                    focal_len_y=focal_len_y,
                    d_thres=d_thres,
                )
                source = o3d.geometry.PointCloud()
                source.points = o3d.utility.Vector3dVector(points)
                source.colors = o3d.utility.Vector3dVector(colors)

                if not _t in transform_matrices:
                    voxel_size = data_cfg.pcm_cfg.voxel_size  # 0.1
                    source_keypoints, source_feature = prepare_dataset(
                        source, voxel_size
                    )
                    target_keypoints, target_feature = prepare_dataset(
                        target, voxel_size
                    )
                    # global_matching
                    result_ransac = execute_global_registration(
                        source_keypoints,
                        target_keypoints,
                        source_feature,
                        target_feature,
                        voxel_size,
                    )
                    # local matching
                    result_icp = refine_registration(
                        source_keypoints, target_keypoints, result_ransac, voxel_size
                    )

                    if result_icp is None:  # Point cloud registration failed.
                        regist_flag = False
                        break
                    else:
                        transform_matrices[_t] = result_icp.transformation

                target = source

                # projection to initial frame
                transform = np.identity(4)
                for _ in sorted(transform_matrices):
                    if _ > _t:
                        break
                    transform = np.dot(transform, transform_matrices[_])

                homogeneous_coords = np.concatenate(
                    [object_coords, np.ones((object_coords.shape[0], 1))], axis=-1
                )
                projected_coords_homogeneous = (transform @ homogeneous_coords.T).T
                projected_coords = (
                    projected_coords_homogeneous[:, :3]
                    / projected_coords_homogeneous[:, 3][:, np.newaxis]
                )

                R_mat = compute_rotation(init_coords, projected_coords[:, :3])
                absolute_rotation = R_mat @ init_rotation
                absolute_position = projected_coords.mean(axis=0)[:3]

            absolute_position *= ratio_depth
            quaternion = R.from_matrix(absolute_rotation).as_quat()
            projected_traj.append(np.concatenate([absolute_position, quaternion]))
        ############################################################################################################
        end = time.time()
        print(f"Time taken for registration: {end - start}", flush=True)

        if not regist_flag:
            continue

        traj_quat = np.stack(projected_traj)

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
            np.save("./viz_data/depth", obs_depth)
            with open("./viz_data/trajectory.pkl", "wb") as f:
                pickle.dump(traj, f)
            with open("./viz_data/text.txt", "w") as f:
                f.write(action_desc)

            traj_rotvec[:, 0] = (
                focal_len_x * traj_rotvec[:, 0] / traj_rotvec[:, 2] + principal_point
            )
            traj_rotvec[:, 1] = (
                focal_len_y * traj_rotvec[:, 1] / traj_rotvec[:, 2] + principal_point
            )

            plt.imshow(image)
            plt.plot(traj_rotvec[:, 0], traj_rotvec[:, 1], c="red")
            plt.savefig("./viz_data/traj.jpg")
            plt.clf()
            import pdb

            pdb.set_trace()
        else:
            os.makedirs(
                f"{args.save_dir}/obs_images/{dataset_name}/{video_uid}", exist_ok=True
            )
            os.makedirs(
                f"{args.save_dir}/depths/{dataset_name}/{video_uid}", exist_ok=True
            )
            os.makedirs(
                f"{args.save_dir}/trajs/{dataset_name}/{video_uid}", exist_ok=True
            )

            pil_image.save(
                f"{args.save_dir}/obs_images/{dataset_name}/{video_uid}/{file_name}.jpg"
            )
            np.save(
                f"{args.save_dir}/depths/{dataset_name}/{video_uid}/{file_name}",
                obs_depth,
            )
            with open(
                f"{args.save_dir}/trajs/{dataset_name}/{video_uid}/{file_name}.pkl",
                "wb",
            ) as f:
                pickle.dump(traj, f)

        # log memory usage
        if idx % 10 == 0:
            log_memory_usage()

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

    args = parser.parse_args()

    main(args)
