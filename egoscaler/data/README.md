## Dataset Overview
The dataset consists of tuples including:
- Action description
- 6DoF object manipulation trajectory
- RGB image
- Depth map
- Colored point cloud (if needed)

Each sample is stored in a structure inspired by the COCO format:

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
- RGB Image: ```f"/path/to/datasetdir/EgoScaler/obs_images/{images[i]['take_name']}/{images[i]['file_name']}.jpg"```
- Depth map: ```f"/path/to/datasetdir/EgoScaler/depths/{images[i]['take_name']}/{images[i]['file_name']}.npy"```
- Colored point cloud: ```f"/path/to/datasetdir/EgoScaler/pcrgbs/{images[i]['take_name']}/{images[i]['file_name']}.npy"```
- Trajectory: ```f"/path/to/datasetdir/EgoScaler/trajs/{images[i]['take_name']}/{images[i]['file_name']}.pickle"```

## Preparation

Currently, our framework supports only the Ego-Exo4D dataset.

- **Ego-Exo4D**: Download [Ego-Exo4D dataset](https://ego-exo4d-data.org/) and follow the official instructions for access approval.
- **Ego4D**: *(Coming soon)*
- **EPIC-Kitchens**: *(Coming soon)*

**NOTE**: To construct evaluation dataset by youself, you also need to download [HOT3D dataset](https://www.projectaria.com/datasets/hot3D/).


## Install dependencies for dataset construction
```bash
# spacy
pip install spacy==3.7.6
python -m spacy download en_core_web_trf

# open3d
conda install -c open3d-admin -c conda-forge open3d==0.18.0
```

2. Install third party
```bash
# Install LLaMA3
cd egoscaler/data/third_party/llama3
pip install -e .

# Install SpaTracker
cd ../SpaTracker
pip install -e .

# Install Depth Anything
cd ../Depth-Anything-V2
pip install -e .
```

## Training Dataset Construction
1. Obtain Candidates
```bash

```


## Evaluation Dataset Construction