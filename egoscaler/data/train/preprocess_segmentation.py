import argparse
import json
import os
import cv2
import supervision as sv
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import glob
import torch
from egoscaler.data.train.tools.grounded_sam2 import GroundedSAM2_VideoTracker


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GroundedSAM2 Video Tracking on a sequence of images."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--monst3r_output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--npz_file_path",
        default="/home/kanazawa/egovision/EgoScaler/egoscaler/data/third_party/mega-sam/outputs_cvd/hoge_sgd_cvd_hr.npz",
    )
    parser.add_argument(
        "--pc_npz_file_path",
        default="/home/kanazawa/egovision/EgoScaler/output_maps.npz",
    )
    parser.add_argument("--save_dir", default="/your/path/to/savedir/EgoScaler")
    parser.add_argument(
        "--vis_segmentation",
        action="store_true",
        help="Whether to visualize segmentation results.",
    )

    return parser.parse_args()


def tracking(args):
    with open(f"{args.save_dir}/infos.json", "r") as f:
        all_data = json.load(f)

    data = all_data[args.index]

    if "dataset_name" in data:
        dataset_name = data["dataset_name"]
    else:
        dataset_name = "hot3d"
    video_uid = data["video_uid"]
    file_name = data.get("file_name", "")

    manipulated_object = data["manipulated_object"]
    device = args.device

    video_tracker = GroundedSAM2_VideoTracker(
        detector_id="IDEA-Research/grounding-dino-base",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        device=device,
    )

    image_dir = f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}"
    if not os.path.exists(image_dir):
        print(f"Image directory {image_dir} does not exist. Skipping.")
        return

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_files:
        print(f"No images found in {image_dir}. Skipping.")
        return

    width = 440
    height = 440
    video_segments, id_to_label_map = video_tracker.track(
        video_path=image_dir,
        text_prompt=manipulated_object,
        box_threshold=0.3,
        text_threshold=0.3,
        resize_wh=(width, height),
    )

    if not video_segments:
        print(f"Tracking failed for {manipulated_object}. Skipping.")
        return

    # Find the target object ID based on the manipulated object
    target_obj_id = None
    for obj_id, label in id_to_label_map.items():
        print(f"Object ID: {obj_id}, Label: {label}")
        if manipulated_object.lower() in label.lower():
            target_obj_id = obj_id
            break

    if target_obj_id is None:
        print(f"Could not find matching object ID for {manipulated_object}. Skipping.")
        return

    # Save the segmented video frames
    if args.vis_segmentation:
        print(f"Visualizing segmentation results for {manipulated_object}...")

        frame_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.endswith(".jpg")
        ]
        if not frame_paths:
            print(f"No frames found in {image_dir}. Skipping video creation.")
            return

        mask_annotator = sv.MaskAnnotator(opacity=0.5)

        annotated_frames = []
        tmp_frame_dir = "tmp_annotated_frames_for_video"
        os.makedirs(tmp_frame_dir, exist_ok=True)

        for frame_idx in tqdm(sorted(video_segments.keys()), desc="Annotating frames"):
            image_full_res = cv2.imread(frame_paths[frame_idx])

            segments_in_frame = video_segments.get(frame_idx, {})

            if segments_in_frame:
                object_ids = list(segments_in_frame.keys())
                masks = np.array(list(segments_in_frame.values()))

                if len(masks) == 0:
                    annotated_frames.append(
                        cv2.cvtColor(image_full_res, cv2.COLOR_BGR2RGB)
                    )
                    continue

                mask_height, mask_width = masks[0].shape

                image_resized = cv2.resize(
                    image_full_res,
                    (mask_width, mask_height),
                    interpolation=cv2.INTER_AREA,
                )

                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks),
                    mask=masks,
                    class_id=np.array(object_ids),
                )

                annotated_frame = mask_annotator.annotate(
                    scene=image_resized, detections=detections
                )
            else:
                annotated_frame = cv2.cvtColor(image_full_res, cv2.COLOR_BGR2RGB)

            output_frame_path = os.path.join(
                tmp_frame_dir, f"frame_{frame_idx:05d}.jpg"
            )
            cv2.imwrite(output_frame_path, annotated_frame)

        annotated_frame_paths = sorted(glob.glob(os.path.join(tmp_frame_dir, "*.jpg")))
        if annotated_frame_paths:
            clip = ImageSequenceClip(annotated_frame_paths, fps=10)
            clip.write_videofile("segmentation_video.mp4", codec="libx264", logger=None)
            print("Visualization video saved to: segmentation_video.mp4")

        if args.vis_segmentation:
            OUTPUT_SIZE = 256

            object_images_dir = os.path.join(args.monst3r_output_dir, "object_images")
            os.makedirs(object_images_dir, exist_ok=True)

            for frame_idx, segments in video_segments.items():
                if target_obj_id in segments:
                    mask = segments[target_obj_id]
                    image_full_res = cv2.imread(frame_paths[frame_idx])
                    mask_height, mask_width = mask.shape

                    image_resized = cv2.resize(
                        image_full_res, (mask_width, mask_height)
                    )

                    ys, xs = np.where(mask > 0)
                    if ys.size == 0 or xs.size == 0:
                        continue

                    y_min, y_max = ys.min(), ys.max()
                    x_min, x_max = xs.min(), xs.max()

                    cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]
                    cropped_image = image_resized[y_min : y_max + 1, x_min : x_max + 1]

                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
                    cropped_image[:, :, 3] = cropped_mask.astype(np.uint8) * 255

                    crop_h, crop_w = cropped_image.shape[:2]
                    canvas_size = max(crop_h, crop_w)
                    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

                    y_offset = (canvas_size - crop_h) // 2
                    x_offset = (canvas_size - crop_w) // 2
                    canvas[
                        y_offset : y_offset + crop_h, x_offset : x_offset + crop_w
                    ] = cropped_image

                    output_image = cv2.resize(
                        canvas, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA
                    )

                    output_image_path = os.path.join(
                        object_images_dir,
                        f"object_{target_obj_id}_frame_{frame_idx:05d}.png",
                    )
                    cv2.imwrite(output_image_path, output_image)

    return video_segments, target_obj_id


def filter_pointcloud_by_mask(scene_pc, scene_colors, object_mask):
    if scene_pc.ndim == 4 and scene_pc.shape[0] == 1:
        scene_pc = scene_pc.squeeze(0)

    if scene_colors.ndim == 4 and scene_colors.shape[0] == 1:
        scene_colors = scene_colors.squeeze(0)

    mask_coords = np.argwhere(object_mask)
    if mask_coords.size == 0:
        return np.array([]), np.array([])  # No points in the mask

    object_pc = scene_pc[mask_coords[:, 0], mask_coords[:, 1]]
    object_colors = scene_colors[mask_coords[:, 0], mask_coords[:, 1]]

    if object_pc.shape[0] == 0:
        print("No valid points found in the object mask.")
        return np.array([]), np.array([])

    print(f"Extracted point cloud shape: {object_pc.shape}")

    # Remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_pc)

    # Perform DBSCAN clustering to find the main object
    # eps: cluster radius
    # min_points: minimum number of points in a cluster
    eps = 0.05
    min_points = 100
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True)
    )

    # Filter out noise points (label -1)
    if np.all(labels == -1):
        print("All points are noise. No clusters found.")
        return np.array([]), np.array([])

    # Find the largest cluster
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        print("Could not find any clusters.")
        object_pc = np.array([])
        object_colors = np.array([])
    else:
        main_cluster_label = unique_labels[counts.argmax()]

        # Extract points belonging to the largest cluster
        indices = np.where(labels == main_cluster_label)[0]
        object_pc = object_pc[indices]
        object_colors = object_colors[indices]

    return object_pc, object_colors


def filter_pointcloud_and_visualize_result(args, video_segments, target_obj_id):
    npz_file_path = args.npz_file_path
    data = np.load(npz_file_path)
    pc_npz_file_path = args.pc_npz_file_path
    pc_data = np.load(pc_npz_file_path)
    depth = data["depths"]  # (N, H, W)
    images = data["images"]  # (N, H, W, 3)
    intrinsic = data["intrinsic"]  # (3, 3)
    cam_c2w = data["cam_c2w"]  # (N, 4, 4)

    point_maps = pc_data["point_maps"]  # (N, H, W, 3)
    color_maps = pc_data["color_maps"]  # (N, H, W, 3)
    scene_pointclouds = point_maps
    colors = color_maps

    object_points_sequence = []
    object_colors_sequence = []

    for frame_idx in range(len(scene_pointclouds)):
        scene_pc_frame = scene_pointclouds[frame_idx]  # (H, W, 3)
        scene_color_frame = colors[frame_idx]  # (H, W, 3)

        if frame_idx in video_segments and target_obj_id in video_segments[frame_idx]:
            print(f"Processing frame {frame_idx} for object ID {target_obj_id}...")
            object_mask = video_segments[frame_idx][target_obj_id]

            print("scene_pc shape:", scene_pc_frame.shape)
            print("object_mask shape:", object_mask.shape)
            object_pc, object_color = filter_pointcloud_by_mask(
                scene_pc_frame, scene_color_frame, object_mask
            )
            object_points_sequence.append(object_pc)
            object_colors_sequence.append(object_color)
        else:
            print(
                f"Warning: No tracking data for frame {frame_idx}. Adding empty point cloud and colors."
            )
            object_points_sequence.append(np.array([]))
            object_colors_sequence.append(np.array([]))

        if args.vis_segmentation:
            if frame_idx == 0:
                os.makedirs(f"{args.monst3r_output_dir}/visualization", exist_ok=True)
                if object_points_sequence[-1].size > 0:
                    scene_pc_reshaped = scene_pc_frame.reshape(-1, 3)
                    scene_color_reshaped = scene_color_frame.reshape(-1, 3)
                    import viser

                    server = viser.ViserServer()
                    server.add_point_cloud(
                        name="scene",
                        points=scene_pc_reshaped,
                        colors=scene_color_reshaped,
                        point_size=0.005,
                    )

                    object_points_sequence_red = np.ones_like(
                        object_points_sequence[-1]
                    ) * [1, 0, 0]
                    server.add_point_cloud(
                        name="object",
                        points=object_points_sequence[-1],
                        colors=object_points_sequence_red,
                        point_size=0.005,
                    )
                    print("Viser server is running at http://localhost:8080")

                    import time

                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("Server stopped.")

    if args.vis_segmentation:
        output_overlay_frame_dir = "./viz_overlay_frames"
        os.makedirs(output_overlay_frame_dir, exist_ok=True)
        overlay_frame_idx = 0

        print("--- Generating overlay visualization frames ---")
        new_width = 440
        new_height = 440
        focal_len_x = intrinsic[0, 0]
        focal_len_y = intrinsic[1, 1]
        principal_point_x = intrinsic[0, 2]
        principal_point_y = intrinsic[1, 2]
        from PIL import Image

        with open(f"{args.save_dir}/infos.json", "r") as f:
            all_data = json.load(f)
        data = all_data[args.index]

        if "dataset_name" in data:
            dataset_name = data["dataset_name"]
        else:
            dataset_name = "hot3d"
        video_uid = data["video_uid"]
        file_name = data.get("file_name", "")
        image_paths = sorted(
            glob.glob(
                f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/*.jpg"
            )
        )
        for i in tqdm(range(len(image_paths)), desc="Visualizing Overlays"):
            current_frame_image = Image.open(image_paths[i]).resize(
                (new_width, new_height)
            )
            R_c2w_frame_i = cam_c2w[i][:3, :3]
            t_c2w_frame_i = cam_c2w[i][:3, 3]

            R_w2c_frame_i = R_c2w_frame_i.T
            t_w2c_frame_i = -R_w2c_frame_i @ t_c2w_frame_i

            focal_x_i = focal_len_x
            focal_y_i = focal_len_y
            pp_x_i = principal_point_x
            pp_y_i = principal_point_y

            current_object_pc = object_points_sequence[i]

            fig, ax = plt.subplots(figsize=(new_width / 100, new_height / 100), dpi=100)

            ax.imshow(np.array(current_frame_image))

            ax.axis("off")

            if len(current_object_pc) > 0:
                object_pc_camera_coords = (
                    R_w2c_frame_i @ current_object_pc.T + t_w2c_frame_i[:, np.newaxis]
                ).T

                valid_pc_mask = object_pc_camera_coords[:, 2] > 0.001

                projected_u = (
                    focal_x_i
                    * object_pc_camera_coords[valid_pc_mask, 0]
                    / object_pc_camera_coords[valid_pc_mask, 2]
                ) + pp_x_i
                projected_v = (
                    focal_y_i
                    * object_pc_camera_coords[valid_pc_mask, 1]
                    / object_pc_camera_coords[valid_pc_mask, 2]
                ) + pp_y_i

                ax.scatter(
                    projected_u, projected_v, s=5, c="red", alpha=0.5, marker="."
                )

            ax.set_xlim(0, new_width)
            ax.set_ylim(new_height, 0)

            plt.savefig(
                os.path.join(
                    output_overlay_frame_dir, f"overlay_{overlay_frame_idx:05d}.png"
                ),
                bbox_inches="tight",
                pad_inches=0,
            )

            plt.clf()
            plt.close(fig)

            overlay_frame_idx += 1

        print(
            f"Generated {overlay_frame_idx} overlay frames in {output_overlay_frame_dir}/"
        )

    # save pointclouds and colors
    save_pointclouds_and_colors(args, object_points_sequence, object_colors_sequence)


def save_pointclouds_and_colors(
    args,
    object_points_sequence,
    object_colors_sequence,
):
    os.makedirs(f"{args.monst3r_output_dir}/object_pointclouds", exist_ok=True)
    os.makedirs(f"{args.monst3r_output_dir}/object_colors", exist_ok=True)

    for frame_idx in range(len(object_points_sequence)):
        if len(object_points_sequence[frame_idx]) == 0:
            continue

        output_pc_path = (
            f"{args.monst3r_output_dir}/object_pointclouds/frame_{frame_idx:05d}.npy"
        )
        output_color_path = (
            f"{args.monst3r_output_dir}/object_colors/frame_{frame_idx:05d}.npy"
        )

        np.save(output_pc_path, object_points_sequence[frame_idx])
        np.save(output_color_path, object_colors_sequence[frame_idx])

    print(
        f"Point clouds and colors saved to {args.monst3r_output_dir}/object_pointclouds and {args.monst3r_output_dir}/object_colors."
    )


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Run tracking
    video_segments, target_obj_id = tracking(args)

    if not video_segments:
        print("No video segments found. Exiting.")
        return

    # Filter point cloud and visualize results
    filter_pointcloud_and_visualize_result(args, video_segments, target_obj_id)

    print(f"Tracking and point cloud filtering completed for {args.index}.")


if __name__ == "__main__":
    args = parse_args()
    args.index = 1  # Set index to 0 for testing purposes
    main(args)
