import os
from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from data_loaders.mano_layer import MANOHandModel
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from data_loaders.loader_object_library import ObjectLibrary
from data_loaders.headsets import Headset
from tqdm import tqdm
from glob import glob
from projectaria_tools.core import data_provider, calibration
import numpy as np
import json
import time
import datetime
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import json
import argparse
import re
from PIL import Image, ImageDraw, ImageFont
from egoscaler.configs import CameraConfig as camera_cfg
import dotenv

dotenv.load_dotenv(dotenv_path=".env")


def price_gpt4o_usd(response):
    price_input_per_1k = 5./1000
    price_output_per_1k = 15./1000
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return input_tokens, output_tokens, round((input_tokens * price_input_per_1k + output_tokens * price_output_per_1k) / 1000, 5)

class AzureGpt4o(object):
    def __init__(self, prompt):
        
        self.prompt = prompt
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __call__(self, pil_frames, manipulated_object_name):
        call_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        base64_frames = self._pil_frames_to_base64(pil_frames)
        response = self.client.chat.completions.create(
            model= 'gpt-4o', #"gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": self.prompt
                },
                {   
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f'Focus on the interaction between the {manipulated_object_name} and the hand.'}
                    ] + [
                        {"type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{base64_frame}",
                             "detail": 'low'},
                        } for base64_frame in base64_frames
                  ]
                }
            ],
            max_tokens=4096,
            temperature=0.1
        )
        execution_time = time.time() - start_time
        
        input_tokens, output_tokens, total_price = price_gpt4o_usd(response)
        print(f"入力トークン数: {input_tokens}, 出力トークン数: {output_tokens}, 合計料金 (USD): ${total_price}", flush=True)

        return {'response': response.choices[0].message.content, 'execution_time': execution_time, 'call_time': call_time}
        
    def _pil_frames_to_base64(self, pil_frames, img_format="jpeg"):
        base64_frames = []
        for pil_frame in pil_frames:
            buffer = BytesIO()
            pil_frame = pil_frame.convert('RGB')
            pil_frame.save(buffer, img_format)
            base64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_frames.append(base64_frame)
        # for base64_frame in base64_frames:
        #     print("len(base64_frame)",len(base64_frame))
        return base64_frames

def main(args):
    with open(args.prompt_path, 'r') as f:
        prompt = f.read()
        
    gpt4 = AzureGpt4o(prompt)
    
    font = ImageFont.truetype(args.font_path, 80)
    
    object_library_path = os.path.join(args.root_dir, "assets")
    object_library = load_object_library(object_library_folderpath=object_library_path)
    sequence_paths = glob(f'{args.root_dir}/P*')
    dataset_name = 'hot3d'

    for sequence_path in tqdm(sequence_paths):
        try:
            hot3d_data_provider = Hot3dDataProvider(
                sequence_folder=sequence_path,
                object_library=object_library,
                mano_hand_model=None,
            )
        except Exception as e:
            print(e)
            continue
        
        # since Meta Quest not contain rgb data, just skipping
        if hot3d_data_provider.get_device_type() != Headset.Aria:
            continue
        
        video_uid = sequence_path.split('/')[-1]
        provider = data_provider.create_vrs_data_provider(f"{sequence_path}/recording.vrs")
        
        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)
        device_calibration = provider.get_device_calibration()
        rgb_camera_calibration = device_calibration.get_camera_calib(camera_label)
        T_device_from_camera = rgb_camera_calibration.get_transform_device_camera()
        pinhole = calibration.get_linear_camera_calibration(
            camera_cfg.devices.aria.pinhole_image_size, 
            camera_cfg.devices.aria.pinhole_image_size, 
            camera_cfg.devices.aria.focal_len, 
            camera_label,
            T_device_from_camera
        )
        time_domain = TimeDomain.TIME_CODE
        option = TimeQueryOptions.CLOSEST
        timestamps: list = provider.get_timestamps_ns(stream_id, time_domain)
        video_start_ns, video_end_ns = timestamps[0], timestamps[-1]
        
        # split whole video into 4 sec segmments
        segments = np.arange(video_start_ns, video_end_ns, 2*camera_cfg.time_window*1e+9)
        
        for start_ns, end_ns in zip(segments, segments[1:]):
            file_name = str(int((end_ns+start_ns) / 2))
            
            sampling_rate = 1 / camera_cfg.fps
            duration = np.arange(start_ns, end_ns, 1e+9 * sampling_rate).astype(int)
            duration = duration[np.round(np.arange(0, len(duration), len(duration)//8))]
            
            if os.path.exists(f"{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json"):
                with open(f"{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json", 'r') as f:
                    data = json.load(f)
            else:
                continue
            
            if 'action_description' in data.keys():
                continue
            
            manipulated_object = data['manipulated_object']
            
            images = []
            for i, ns in enumerate(duration):
                pil_image = Image.open(f"{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{round(ns, 3)}.jpg")
                draw = ImageDraw.Draw(pil_image)
                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                position = ((pil_image.width - text_width) // 2, pil_image.height - text_height - 100)  
                draw.text(position, text, font=font, fill="white")  # Change 'fill' to the desired text color
                images.append(pil_image)
                
            try:
                output = gpt4(pil_frames=images, manipulated_object_name=manipulated_object)
                if output['response'].lower() == 'invalid':
                    desc, start_sec, end_sec = None, None, None
                else:
                    desc, start_idx, end_idx = output['response'].split('\n')
                    desc = re.sub('Description: ', '', desc).lower()
                    desc = re.sub('_', ' ', desc)
                    start_idx = int(re.sub('start frame: ', '', start_idx))
                    end_idx = int(re.sub('end frame: ', '', end_idx))
                    start_ns = int(duration[start_idx])
                    end_ns = int(duration[end_idx])
                    start_sec = start_ns * 1e-9
                    end_sec = end_ns * 1e-9
                
            except Exception as e:
                print(e, flush=True)
                continue
            
            if args.visualize:
                resize_size = (704, 704)
                num_columns = (len(images) + 1) // 2
                dst_width = resize_size[0] * num_columns
                dst_height = resize_size[1] * 2 + 500  
                dst = Image.new('RGB', (dst_width, dst_height))
                for idx, image in enumerate(images):
                    image = image.resize(resize_size)
                    row = idx // num_columns
                    col = idx % num_columns
                    dst.paste(image, (resize_size[0] * col, resize_size[1] * row))
                    
                dst.save('temp.jpg')
                import pdb; pdb.set_trace()
                
                data['start_sec'] = start_sec
                data['end_sec'] = end_sec
                data['action_description'] = desc
                
                with open(f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json', 'w') as f:
                    json.dump(data, f)
                
            else:
                data['start_sec'] = start_sec
                data['end_sec'] = end_sec
                data['action_description'] = desc
                
                with open(f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json', 'w') as f:
                    json.dump(data, f)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # data dirs
    parser.add_argument("--root_dir", default="/data/g-liat/yoshida/Hot3D")
    parser.add_argument("--prompt_path", default='/home/yoshida/EgoScaler/egoscaler/data/prompt/get_desc_and_timestamp.txt')
    parser.add_argument("--font_path", default='/home/yoshida/ULAT/data/Menlo-Regular.ttf')
    parser.add_argument("--save_dir", default='/data/g-liat/yoshida/EgoScaler')
    parser.add_argument("--visualize", action='store_true')
    
    args = parser.parse_args()
    
    main(args)