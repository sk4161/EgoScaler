import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import time

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
    
def annotate(image: Union[Image.Image, np.ndarray], detection_results: List[DetectionResult]) -> np.ndarray:
    # Convert PIL Image to OpenCV format
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    # Iterate over detections and add bounding boxes and masks
    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        # Sample a random color for each detection
        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # If mask is available, apply it
        if mask is not None:
            # Convert mask to uint8
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

def plot_detections(
    image: Union[Image.Image, np.ndarray],
    detections: List[DetectionResult],
    save_name: Optional[str] = None
) -> None:
    annotated_image = annotate(image, detections)
    plt.imshow(annotated_image)
    plt.axis('off')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')


class GroundedSAM:
    def __init__(self, detector_id, segmenter_id, device):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.device = device
        
        print("Loading grounding DINO...")
        self.object_detector = pipeline(
            model=self.detector_id, 
            task="zero-shot-object-detection", 
            device=self.device
        )
        print("Loaded.")
        
        print("Loading SAM...")
        self.segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        self.segmentor_processor = AutoProcessor.from_pretrained(self.segmenter_id)
        print("Loaded.")
        
    def get_boxes(self, results: DetectionResult) -> List[List[List[float]]]:
        boxes = []
        for result in results:
            xyxy = result.box.xyxy
            boxes.append(xyxy)

        return [boxes]

    def refine_masks(self, masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
        masks = masks.cpu().float()
        masks = masks.permute(0, 2, 3, 1)
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        if polygon_refinement:
            for idx, mask in enumerate(masks):
                shape = mask.shape
                polygon = self.mask_to_polygon(mask)
                mask = self.polygon_to_mask(polygon, shape)
                masks[idx] = mask

        return masks

    def polygon_to_mask(self, polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert a polygon to a segmentation mask.

        Args:
        - polygon (list): List of (x, y) coordinates representing the vertices of the polygon.
        - image_shape (tuple): Shape of the image (height, width) for the mask.

        Returns:
        - np.ndarray: Segmentation mask with the polygon filled.
        """
        # Create an empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)

        # Convert polygon to an array of points
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

        return mask
    
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[int]]:
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract the vertices of the contour
        polygon = largest_contour.reshape(-1, 2).tolist()

        return polygon
    
    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        labels: List[str],
        threshold: float = 0.3,
        detector_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
        """
        labels = [label if label.endswith(".") else label+"." for label in labels]
        results = self.object_detector(image, candidate_labels=labels, threshold=threshold) 
        results = [DetectionResult.from_dict(result) for result in results]

        return results
    
    @torch.no_grad()  
    def segment(
        self,
        image: Image.Image,
        detection_results: List[Dict[str, Any]],
        polygon_refinement: bool = False,
        segmenter_id: Optional[str] = None
    ) -> List[DetectionResult]:
        """
        Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
        """

        boxes = self.get_boxes(detection_results)
        inputs = self.segmentor_processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.segmentor_processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        masks = self.refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results

    def predict(
        self, 
        image: Image.Image, 
        labels: List[str],
        threshold: float,
        polygon_refinement: bool=True
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        
        detections = self.detect(image, labels, threshold)
        
        if not len(detections):
            # when no reffered objects in the image
            height, width = image.size
            return np.zeros([1, height, width]), np.zeros([1, 4]), None
        
        detections = self.segment(image, detections, polygon_refinement)

        masks = np.stack([d.mask for d in detections])
        boxes = np.stack([d.box for d in detections])
        scores = np.array([d.score for d in detections])

        return masks, boxes, scores
        
"""
if __name__ == "__main__":

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    labels = ["a cat.", "a remote control."]
    threshold = 0.3

    detector_id = "IDEA-Research/grounding-dino-base"
    segmenter_id = "facebook/sam-vit-base"
    
    grounded_sam = GroundedSAM(detector_id, segmenter_id, device='cuda')
    
    detections = grounded_sam.predict(
        image=image,
        labels=labels,
        threshold=threshold
    )
        
    plot_detections(image_array, detections, "cute_cats.png")
"""