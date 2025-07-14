import os
import json
from tqdm import tqdm
import argparse
import glob
import sys

from PIL import Image
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from egoscaler.configs.camera import CameraConfig as camera_cfg

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def _get_bounding_box(box: "torch.Tensor"):
    xmin, ymin, xmax, ymax = box.int().tolist()
    bbox = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }
    return bbox

class LazyDataset(Dataset):
    def __init__(self, args, dataset, processor):
        self.args = args
        self.dataset = dataset
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):
        images, sizes, queries, durations, dataset_names, video_uids, file_names = list(zip(*batch))
        
        images = [img for batch in images for img in batch]
        queries = [query for batch in queries for query in batch]
        inputs = self.processor(images=images, 
                                text=queries, 
                                padding=True,
                                return_tensors='pt')
        
        return inputs, sizes, durations, dataset_names, video_uids, file_names
    
    def __getitem__(self, item):
        data = self.dataset[item]
        if data["dataset_name"]:
            dataset_name = data["dataset_name"]
        else:
            dataset_name = "hot3d"
        video_uid = data['video_uid']
        file_name = data['file_name']
        manipulated_object = data['manipulated_object']
        
        with open(f"{self.args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json", 'r') as f:
            data = json.load(f)
        
        sampling_rate = 1 / camera_cfg.fps
        timestamp = data['timestamp']
        start_sec = data['start_sec']
        end_sec = data['end_sec']
        
        original_duration = np.round(
            np.arange(
                timestamp-camera_cfg.time_window, 
                timestamp+camera_cfg.time_window, 
                sampling_rate
            ), 
            3
        )
        start_index = np.where(original_duration == round(start_sec, 3))[0]
        end_index = np.where(original_duration == round(end_sec, 3))[0]
        duration = original_duration[start_index[0]:end_index[0] + 1]
        
        images = []
        sizes = []
        queries = []
        image_dir = (
            f"{self.args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}"
        )
        all_image_paths = sorted(glob.glob(f"{image_dir}/*.jpg"))

        duration_from_files = []
        images_to_process = []

        for img_path in all_image_paths:
            try:
                timestamp_int_str = os.path.basename(img_path).split(".")[0]
                timestamp_sec = (
                    float(timestamp_int_str) / 1e9
                )

                if start_sec <= timestamp_sec <= end_sec:
                    images_to_process.append(img_path)
                    duration_from_files.append(round(timestamp_sec, 3))
            except (ValueError, IndexError):
                print(f"Error processing file: {img_path}. Skipping.")
                continue

        if not images_to_process:
            print(f"No valid images found for: {dataset_name}, {video_uid}, {file_name}.")

            return [], [], [], [], dataset_name, video_uid, file_name

        for img_path in images_to_process:
            try:
                pil_image = Image.open(img_path)
            except:
                print(f"Error opening file: {dataset_name}, {video_uid}, {file_name}.")
                sys.exit(1)  # スクリプトを終了
            images.append(pil_image)
            sizes.append(pil_image.size[::-1])
            text = '. '.join(['person', 'hand', manipulated_object])   
            text += '.' 
            queries.append(text)

        duration = np.array(duration_from_files)

        return images, sizes, queries, duration, dataset_name, video_uid, file_name

@torch.no_grad()
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model setting
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # data setting
    with open(f"{args.data_dir}/infos.json", 'r') as f:
        all_data = json.load(f)

    already = set()
    video_uids = glob.glob(f'{args.save_dir}/bboxes/*/*')
    for video_uid in video_uids:
        files = glob.glob(f'{video_uid}/*')
        already.update(files)
    
    print("Removing existing files...")
    infos_path = os.path.join(args.save_dir, "infos")
    bboxes_path = os.path.join(args.save_dir, "bboxes")
    filtered_data = []
    for data in tqdm(all_data, desc="Removing..."):
        if data["dataset_name"]:
            dataset_name = data["dataset_name"]
        else:
            dataset_name = "hot3d"
        video_uid = data['video_uid']
        file_name = data['file_name']
        info_file = os.path.join(infos_path, dataset_name, video_uid, f"{file_name}.json")
        if not os.path.exists(info_file):
            continue
        with open(info_file, 'r') as f:
            info = json.load(f)
        if info['start_sec'] is None:
            continue
        bbox_file = os.path.join(bboxes_path, dataset_name, video_uid, f"{file_name}.json")
        if bbox_file not in already:
            filtered_data.append(data)
        
    dataset = LazyDataset(
        args,
        filtered_data, 
        processor
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=6, pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    
    for inputs, sizes, durations, dataset_names, video_uids, file_names in tqdm(dataloader):

        inputs = inputs.to(device)
        sizes = [size for batch in sizes for size in batch]
        duration_length = [len(duration) for duration in durations]
        durations = [duration for batch in durations for duration in batch] 

        outputs = model(**inputs)

        outputs = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=sizes 
        )

        d_start = 0
        # Process outputs and save results in a single loop
        for d_end, dataset_name, video_uid, file_name in zip(duration_length, dataset_names, video_uids, file_names):
            duration = durations[d_start:d_start + d_end]
            output_slice = outputs[d_start:d_start + d_end]
            d_start += d_end

            instance = {}
            for _t, output in zip(duration, output_slice):
                instance_results = [
                    {"score": score.item(), "label": label, "box": _get_bounding_box(box)}
                    for score, label, box in zip(output["scores"], output["labels"], output["boxes"])
                    if score.item() > 0
                ]
                instance[round(_t, 3)] = instance_results
                
            os.makedirs(f'{args.save_dir}/bboxes/{dataset_name}/{video_uid}', exist_ok=True)
            
            with open(f'{args.save_dir}/bboxes/{dataset_name}/{video_uid}/{file_name}.json', 'w') as f:
                json.dump(instance, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data dirs
    parser.add_argument("--data_dir", default='./data')
    parser.add_argument("--save_dir", default='/your/path/to/savedir/EgoScaler')

    parser.add_argument("--batch_size", type=int, default=4)
    
    args = parser.parse_args()
    
    main(args)
