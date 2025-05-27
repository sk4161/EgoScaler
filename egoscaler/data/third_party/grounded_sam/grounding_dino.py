import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))


class GroundingDINO:
    def __init__(self, detector_id, device):
        self.detector_id = detector_id
        self.device = device
        
        print("Loading grounding DINO...")
        self.object_detector = pipeline(model=self.detector_id, 
                                   task="zero-shot-object-detection", 
                                   device=self.device)
        print("Loaded.")
    
    @torch.no_grad()
    def detect(
        self,
        inputs,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        results = self.object_detector(inputs, threshold=threshold)
        results = [[DetectionResult.from_dict(res) for res in result] for result in results]
        return results

    def predict(
        self, 
        images: List[Image.Image], 
        labels: List[List[str]],
        threshold: float,
        polygon_refinement: bool=True
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        
        inputs = []
        for i in range(len(images)):
            inputs.append(
                {
                    'image': images[i],
                    'candidate_labels': [label if label.endswith(".") else label+"." for label in labels[i]]
                }
            )
        
        detections = self.detect(inputs, threshold)

        return detections