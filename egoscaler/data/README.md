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

3. (Option) Install hand-object-detector

We used [hand_object_detector](https://github.com/ddshan/hand_object_detector) to improve detecting correct manipulated object.
However, since this detector depends on old packages, we build another conda env for this.
```bash
(working under hand_object_detector directory)

conda create --name handobj_new python=3.8
conda activate handobj_new
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Compile the cuda dependencies 
pip install -r requirements.txt
cd lib
python setup.py build develop
```

4. Download weights 
- For SpaTracker: We use this [checkpoint](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ) and place it under ```SpaTracker/checkpoints```.
- For Depth Anything: We use indoor metric depth [checkpotint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth?download=true) and place it under ```Depth-Anything-V2/checkpoints```.
- For LLaMA3: Please follow officual instruction in [huggingface](https://huggingface.co/meta-llama).
- (Option) For hand_object_detector: We use [egocentric version](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) and place it under ```hand_object_detector/models/res101_handobj_100K/pascal_voc```.

## Training Dataset Construction
1. **Obtain Candidates**: Extracting candidates of dataset instances and format basic information in unified mannar.
```bash
bash scripts/1_get_cands.sh
```

2. **Filter Candidates**: Filter out unsuitable candidates (e.g., talking/walking scenarios) using both rule-based and LLaMA3-70B-Instruct-based methods.

```bash
# Adjust --nproc_per_node and batch size according to your environment
bash scripts/2_filter_cands.sh
```

3. **Obtain Manipulated Object Name**: Obtain target object (manipulated object by camera wearer's hands) using LLaMA3-70B-Instruct.

```bash
bash scripts/3_get_object_name.sh 
```

4. **Pre-process images**: For faster processing of extracting trajectories, we pre-process videos to images in advance.

```bash
bash scripts/4_get_images.sh
```

5. **Conduct Temporal Action Localization**: Obtain action start/end timestamp using ChatGPT API.

We used GPT-4o.

```bash
export AZURE_OPENAI_KEY=...
export AZURE_OPENAI_ENDPOINT=...

bash scripts/5_get_timestamps.sh

# After processing all data
python egoscaler/data/train/5_get_timestamp.py --format_all
# -> generating infos.json under data dir
```

6. **Detect Humans / Hands**: To improve point cloud registration accuracy, we use human / hand bouding boxes to remove moving objects.

```bash
bash scripts/6_get_bouding_box.py
```

(Option) If you want to obtain better trajectories, I recommend you to conduct before moving to step 7.
```bash
python prepro_for_EgoScaler.py
```

7. **Extract 6DoF Object Trajectories**: Tracking active objects and extracting the 6DoF trajectories.

```bash
bash scripts/7_get_object_trajectory.sh
```


## Evaluation Dataset Construction

```bash
cd egoscaler/data/eval
```

```bash
# Pre-processing images 
python 1_get_image.py 

# target object is determined by total travel distance
python 2_get_manipulated_object.py

# obtain start/end timestamp and action description 
python 3_get_desc_timestamp.py

# extract object manipulation trajectory
# by using recorded object poses
python 4_get_object_trajecotry.py
```