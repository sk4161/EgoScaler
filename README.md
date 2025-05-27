# Generating 6DoF Object Manipulation Trajectories from Action Description in Egocentric Vision

### Dataset

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

### 🛠️ Install
1. Create conda env
```bash
# Python version 3.8 or higher is required
conda create -n egoscaler python=3.8.17
conda activate egoscaler
pip install -e . (under root EgoScaler directory)
```

2. Install pytorch, torchvision
```bash
# Experiments were conducted using CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install open3d
```

3. Install dependencies
```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

3. Install third party library
```bash
# Install LLaMA3
cd egoscaler/data/third_party

git clone https://github.com/meta-llama/llama3.git
cd llama3
pip install -e .
```


4. Install dependencies for dataset construction
Follow [here](./data/README.md).

### Training/Evaluation Dataset Construction
Follow [here](./data/README.md).