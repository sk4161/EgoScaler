import os
import json
import pickle
from PIL import Image
import argparse
from hod import HOD
import numpy as np
from tqdm import tqdm

def get_bbox(obj_dets, hand_dets):
    o_bboxes = []
    if obj_dets is not None:
        for i in range(obj_dets.shape[0]):
            o_bboxes.append(list(int(np.round(x)) for x in obj_dets[i, :4]))
    h_bboxes = []
    if hand_dets is not None:    
        for i in range(hand_dets.shape[0]):
            h_bboxes.append(list(int(np.round(x)) for x in hand_dets[i, :4]))
            h_bboxes[i].append(hand_dets[i, 5]) # state
            h_bboxes[i].append(hand_dets[i, -1]) # l or r
    return o_bboxes, h_bboxes

def main(args):    
    
    hod = HOD(args)
    
    with open(f'{args.data_dir}/infos.json', 'r') as f:
        all_data = json.load(f)
    
    # for multi processing
    print(f"{args.start_index} to {args.end_index}.")
    if args.start_index == 0 and args.end_index == -1:
        pass
    elif args.end_index >= len(all_data) or args.end_index == -1:
        all_data = all_data[args.start_index:]
    else:
        all_data = all_data[args.start_index:args.end_index]
        
    for data in tqdm(all_data):
        dataset_name = data['dataset_name']
        video_uid = data['video_uid']
        take_name = data.get('take_name', '')
        vrs_file_name = data.get('vrs_file_name', '')
        file_name = data.get('file_name', '')  # もし必要なら
        
        if os.path.exists(f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json'):
            with open(f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json', 'r') as f:
                data = json.load(f)
            if data['start_sec'] is None:
                continue
        else:
            continue
        
        if os.path.exists(f'{args.save_dir}/hods/{dataset_name}/{video_uid}/{file_name}.pkl'):
            continue
        
        sampling_rate = 1 / 20.0 # 3 * 1/fps
        timestamp = data['timestamp']
        start_sec = data['start_sec']
        end_sec = data['end_sec']
        
        original_duration = np.round(
            np.arange(
                timestamp-2.0, 
                timestamp+2.0, 
                sampling_rate
            ), 
            3
        )
        start_index = np.where(original_duration == round(start_sec, 3))[0]
        end_index = np.where(original_duration == round(end_sec, 3))[0]
        duration = original_duration[start_index[0]:end_index[0] + 1]
        
        hod_results = {}
        for _t in duration: 
            pil_image = Image.open(f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{round(_t, 3)}.jpg')
            image = np.array(pil_image)
            
            obj_dets, hand_dets, theres_obj, theres_hand, im2show = hod.detect(image)
            o_bboxes, h_bboxes = get_bbox(obj_dets, hand_dets)
                            
            hod_results[_t] = {'obj-bbox': o_bboxes, 'hand-bbox': h_bboxes} 
        
        os.makedirs(f'{args.save_dir}/hods/{dataset_name}/{video_uid}', exist_ok=True)    
        with open(f'{args.save_dir}/hods/{dataset_name}/{video_uid}/{file_name}.pkl', 'wb') as f:
            pickle.dump(hod_results, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda', 
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=132028, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=False)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.8,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.8,
                        type=float,
                        required=False)
        
    # data dirs
    parser.add_argument("--data_dir", default='./data')
    parser.add_argument("--save_dir", default='/your/path/to/savedir/EgoScaler')
    parser.add_argument("--visualize", action='store_true')
    
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    
    args = parser.parse_args()
    
    main(args)