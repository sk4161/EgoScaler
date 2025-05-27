import os
import json
import re
import argparse
from glob import glob
from tqdm import tqdm

from egoscaler.data.utils import egoexo4d_utils  # other datasets are commented for future support
from egoscaler.data.tools import (
    extract_verb_obj, lemmatize_description, format_description
)

DATASET_MODULES = {
    'egoexo4d': egoexo4d_utils,
    # 'ego4d': ego4d_utils,
    # 'epic_kitchens': epic_kitchens_utils,
}

USABLE_SCENARIO = {
    'egoexo4d': ['Cooking', 'Bike Repair', 'Music', 'Health'],
    # 'ego4d': [...],
    # 'epic_kitchens': [...],
}

def load_annotations(dataset_name, split, args):
    return DATASET_MODULES[dataset_name].load_annotations(split, args)

def process_take(dataset_name, take, descriptions):
    return DATASET_MODULES[dataset_name].process_take(take, descriptions)

def process_description(dataset_name, desc_info):
    return DATASET_MODULES[dataset_name].process_description(desc_info)

def main(args):
    # If formatting all, list all JSON candidates (currently unused)
    if args.format_all:
        all_cands_file = glob(f'{args.save_dir}/cands/*/*/*.json')
        return

    candidates = []

    for dataset_name in USABLE_SCENARIO:
        if args.dataset_name != dataset_name:
            continue

        print(f"Processing {dataset_name} dataset...")

        for split in ['train', 'val']:
            descriptions, takes = load_annotations(dataset_name, split, args)

            for take in tqdm(takes, desc=f"Processing {dataset_name}"):
                video_uid, task_name, desc_infos = process_take(dataset_name, take, descriptions)

                if task_name not in USABLE_SCENARIO[dataset_name]:
                    continue

                for desc_info in desc_infos:
                    raw_desc, timestamp, subject, ego_visible, unsure, not_interaction = \
                        process_description(dataset_name, desc_info)

                    if unsure or subject != 'C' or not ego_visible or not_interaction:
                        continue

                    file_name = f"{video_uid}_{round(timestamp, 3)}"
                    cand_path = f'{args.save_dir}/cands/{dataset_name}/{video_uid}/{file_name}.json'

                    if os.path.exists(cand_path):
                        continue

                    desc = format_description(raw_desc)
                    lemma_desc = lemmatize_description(desc)
                    _verb, _object = extract_verb_obj(lemma_desc)
                    action_description = re.sub(r'\s+\.', '.', re.sub('c ', '', lemma_desc))

                    if _verb is None or _object is None:
                        continue

                    instance = {
                        'dataset_name': dataset_name,
                        'video_uid': video_uid,
                        'take_name': take.get('take_name', None),
                        'vrs_file_name': take.get('capture', {}).get('cameras', [{}])[0].get('cam_id', None),
                        'timestamp': timestamp,
                        'raw_description': desc,
                        'lemma_description': lemma_desc,
                        'action_description': action_description,
                        'verb': _verb,
                        'object': _object,
                        'task_name': task_name,
                        'file_name': file_name
                    }

                    candidates.append(instance)
                    os.makedirs(os.path.dirname(cand_path), exist_ok=True)
                    with open(cand_path, 'w') as f:
                        json.dump(instance, f)

    print(f"Total candidates: {len(candidates)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_egoexo4d_dir", default='/your/path/to/egoexo4d')
    parser.add_argument("--root_ego4d_dir", default='/your/path/to/ego4d')
    parser.add_argument("--root_epic_kitchens_dir", default='/your/path/to/epic-kitchens')
    parser.add_argument("--save_dir", default='/your/path/to/savedir/EgoScaler/')
    parser.add_argument("--dataset_name", type=str, default='egoexo4d',
                        choices=['egoexo4d', 'ego4d', 'epic_kitchens'])
    parser.add_argument("--format_all", action="store_true")
    args = parser.parse_args()
    main(args)
