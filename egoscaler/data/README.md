# EgoScaler: Generating 6DoF Object Manipulation Trajectories from Action Description in Egocentric Vision.

1. Install dependencies
```bash
# spacy
pip install spacy==3.7.6
python -m spacy download en_core_web_trf

# open3d
conda install -c open3d-admin -c conda-forge open3d==0.18.0


```

2. Install third party
```bash

```

### Dataset Overview

The EgoTraj dataset consists of nearly 30K tuples of action descriptions, 6DoF object manipulation trajectories, depth maps, RGB images, and point clouds.

We format our dataset following COCO format:

```json
      {
            "images":[
                  {
                        "file_name": "93a6cc04-f932-4555-93de-bf55a089ac80_848.198", 
                        "take_name": "fair_cooking_07_2", 
                        "id": 704
                  },
            ],
            "annotations":[
                  {
                        "image_id": 704, 
                        "id": 704, 
                        "caption": "C holds the onion with both hands.", 
                        "lemma_caption": "c hold the onion with both hand ."
                  }
            ]
      }
```

To access each trajectory, depth map, RGB image, and point cloud:
- image: ```f"/path/to/exoegotraj/obs_images/{images[i]['take_name']}/{images[i]['file_name']}.jpg"```
- depth map: ```f"/path/to/exoegotraj/depths/{images[i]['take_name']}/{images[i]['file_name']}.npy"```
- colored point cloud: ```f"/path/to/exoegotraj/pcrgbs/{images[i]['take_name']}/{images[i]['file_name']}.npy"```
- trajectory: ```f"/path/to/exoegotraj/trajs/{images[i]['take_name']}/{images[i]['file_name']}.pickle"```


3. Install third party library
```bash
# Install LLaMA3
cd egoscaler/data/third_party

git clone https://github.com/meta-llama/llama3.git
cd llama3
pip install -e .
```