import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pickle
import json

class DatasetBase(Dataset):
    """
    Base class for datasets used in EgoScaler.
    Provides common functionality for loading and processing data.
    """
    def __init__(self, args, split: str):
        """
        Initialize the DatasetBase class.

        Parameters:
        - args: Arguments containing dataset configurations.
        - split (str): Dataset split (e.g., 'train', 'val', 'test').
        """
        super().__init__()
        self.args = args
        self.root_dir = args.root_dir
        self.split = split

        # Initialize dataset-specific attributes
        self.num_steps = None
        self.action_dim = None

        # Load dataset based on the split
        if split in ["train", "val", "test"]:
            with open(f"{args.data_dir}/{split}.json", 'r') as f:
                dataset = json.load(f)
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

        # Map image IDs to data and load annotations
        self.id2data = {item['id']: item for item in dataset['images']}
        self.annotations = dataset['annotations']

    def __len__(self) -> int:
        """
        Return the number of annotations in the dataset.

        Returns:
        - int: Number of annotations.
        """
        return len(self.annotations)

    def collate_fn(self, batch: list) -> dict:
        """
        Collate function for batching data.

        Parameters:
        - batch (list): List of data samples.

        Returns:
        - dict: Batched data.
        """
        image_ids, images, descs, trajs = zip(*batch)
        return {
            'image_ids': image_ids,
            'images': images,
            'action_descriptions': descs,
            'trajectories': trajs
        }

    def __getitem__(self, item: int):
        """
        Retrieve a single data sample.

        Parameters:
        - item (int): Index of the data sample.

        Returns:
        - tuple: (image_id, image, description, trajectory).
        """
        annot = self.annotations[item]
        image_id = annot['image_id']
        data = self.id2data[image_id]

        dataset_name = data['dataset_name']
        video_uid = data['video_uid']
        file_name = data['file_name']

        # Process action description
        desc = annot['action_description']
        try:
            desc = desc.lower()
        except AttributeError:
            print(f"Error processing description for {dataset_name}, {video_uid}, {file_name}")

        # Load image
        image_path = f"{self.root_dir}/obs_images/{dataset_name}/{video_uid}/{file_name}.jpg"
        pil_image = Image.open(image_path)
        image = np.array(pil_image)

        # Load trajectory
        traj_path = f"{self.root_dir}/trajs/{dataset_name}/{video_uid}/{file_name}.pkl"
        with open(traj_path, 'rb') as f:
            traj_info = pickle.load(f)
        traj = traj_info['traj_rotvec']

        return torch.tensor(image_id), pil_image, desc, traj