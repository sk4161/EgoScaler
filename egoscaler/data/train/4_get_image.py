import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from moviepy.editor import VideoFileClip

from projectaria_tools.core import data_provider
from egoscaler.data.utils import egoexo4d_utils

DATASET_MODULES = {
    'egoexo4d': egoexo4d_utils,
    # 'ego4d': ego4d_utils,
    # 'epic_kitchens': epic_kitchens_utils
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )

def extract_images(dataset_name, provider, data, save_path, video_duration):
    """
    Dispatch to the dataset-specific image extraction function.
    """
    return DATASET_MODULES[dataset_name].extract_images(
        provider, data, save_path, video_duration
    )

def load_data(save_dir, start_index, end_index):
    all_infos_file = glob(f"{save_dir}/infos/*/*/*.json")
    all_data = []
    for file_name in tqdm(all_infos_file, desc="Loading data files"):
        with open(file_name, 'r') as f:
            data = json.load(f)
        all_data.append(data)

    total_data = len(all_data)
    logging.info(f"Total data points: {total_data}")

    if start_index < 0 or start_index >= total_data:
        logging.error("Invalid start_index.")
        sys.exit(1)

    if end_index == -1 or end_index > total_data:
        end_index = total_data

    subset_data = all_data[start_index:end_index]
    logging.info(f"Processing data from index {start_index} to {end_index} (total: {len(subset_data)})")
    return subset_data

def group_data_by_video_uid(all_data):
    nested_data = defaultdict(list)
    for data in all_data:
        video_uid = data['video_uid']
        nested_data[video_uid].append(data)
    return nested_data

def process_take(args, take_data):
    dataset_name = take_data[0]['dataset_name']
    video_uid = take_data[0]['video_uid']
    take_name = take_data[0].get('take_name', '')
    vrs_file_name = take_data[0].get('vrs_file_name', '')

    save_dir = os.path.join(args.save_dir, 'images', dataset_name, video_uid)
    os.makedirs(save_dir, exist_ok=True)

    if dataset_name == 'egoexo4d':
        vrs_path = os.path.join(args.root_egoexo4d_dir, 'takes', take_name, f"{vrs_file_name}.vrs")
        try:
            provider = data_provider.create_vrs_data_provider(vrs_path)
        except Exception as e:
            logging.error(f"Failed to create data provider for {vrs_path}: {e}")
            return

        for data in take_data:
            individual_save_path = os.path.join(save_dir, data['file_name'])
            extract_images(dataset_name, provider, data, individual_save_path, None)

    elif dataset_name in ['ego4d', 'epic_kitchens']:
        if dataset_name == 'ego4d':
            video_path = os.path.join(args.root_ego4d_dir, 'v1', 'full_scale', f"{video_uid}.mp4")
        else:
            video_dir = video_uid.split('_')[0]
            video_path = os.path.join(args.root_epic_dir, video_dir, 'videos', f"{video_uid}.MP4")

        if not os.path.exists(video_path):
            logging.error(f"Video file does not exist: {video_path}")
            return

        try:
            with VideoFileClip(video_path) as video:
                video_duration = video.duration
        except Exception as e:
            logging.error(f"Failed to open video file {video_path}: {e}")
            return

        for data in take_data:
            individual_save_path = os.path.join(save_dir, data['file_name'])
            extract_images(dataset_name, video_path, data, individual_save_path, video_duration)

    else:
        logging.warning(f"Unknown dataset name: {dataset_name}")

def main(args):
    subset_data = load_data(args.save_dir, args.start_index, args.end_index)
    nested_data = group_data_by_video_uid(subset_data)

    progress = tqdm(total=len(nested_data), desc="Processing takes")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_take = {
            executor.submit(process_take, args, take_data): take_uid
            for take_uid, take_data in nested_data.items()
        }

        for future in as_completed(future_to_take):
            take_uid = future_to_take[future]
            try:
                future.result()
                logging.info(f"Completed processing for video_uid: {take_uid}")
            except Exception as e:
                logging.error(f"Error processing video_uid {take_uid}: {e}")
            progress.update(1)

    progress.close()
    logging.info("All processing completed.")

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="Process video datasets and extract frames/images.")
    parser.add_argument("--root_egoexo4d_dir", default='/your/path/to/egoexo4d', help="Root directory for egoexo4d dataset")
    parser.add_argument("--root_ego4d_dir", default='/your/path/to/ego4d', help="Root directory for ego4d dataset")
    parser.add_argument("--root_epic_dir", default='/your/path/to/epic-kitchens', help="Root directory for epic_kitchens dataset")
    parser.add_argument("--save_dir", default='/your/path/to/savedir/EgoScaler', help="Directory to save extracted images")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for processing data")
    parser.add_argument("--end_index", type=int, default=-1, help="End index for processing data (-1 for all)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker threads for processing")

    args = parser.parse_args()
    main(args)