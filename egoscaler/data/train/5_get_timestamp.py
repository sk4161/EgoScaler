import os
import time
import datetime
import json
import base64
import re
import argparse
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from moviepy.editor import ImageSequenceClip
from openai import AzureOpenAI
from egoscaler.configs import CameraConfig as camera_cfg

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def price_gpt4o_usd(response):
    price_input_per_1k = 5./1000
    price_output_per_1k = 15./1000
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return input_tokens, output_tokens, round((input_tokens * price_input_per_1k + output_tokens * price_output_per_1k) / 1000, 5)

class AzureGpt4o:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def __call__(self, query: str, active_object: str, pil_frames: List[Image.Image]) -> Dict[str, Any]:
        call_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_time = time.time()
        base64_frames = self._pil_frames_to_base64(pil_frames)
        
        # メッセージの準備
        messages = [
            {
                "role": "system",
                "content": self.prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"action description: {query}, manipulated object: {active_object}"
                    }
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_frame}",
                            "detail": 'low'
                        }
                    } for base64_frame in base64_frames
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model='gpt-4o', 
            messages=messages,
            max_tokens=4096,
            temperature=0.1
        )
        execution_time = time.time() - start_time

        input_tokens, output_tokens, total_price = price_gpt4o_usd(response)

        return {
            'response': response.choices[0].message.content,
            'execution_time': execution_time,
            'call_time': call_time,
            'total_price': total_price 
        }

    def _pil_frames_to_base64(self, pil_frames: List[Image.Image], img_format: str = "jpeg") -> List[str]:
        base64_frames = []
        for pil_frame in pil_frames:
            buffer = BytesIO()
            pil_frame = pil_frame.convert('RGB')
            pil_frame.save(buffer, format=img_format)
            base64_frame = base64.b64encode(buffer.getvalue()).decode("utf-8")
            base64_frames.append(base64_frame)
        return base64_frames

def main(args):

    with open(args.prompt_path, 'r') as f:
        prompt = f.read()
    gpt4 = AzureGpt4o(prompt)

    font = ImageFont.truetype('./Menlo-Regular.ttf', 80)

    with open(f'{args.data_dir}/infos.json', 'r') as f:
        all_data = json.load(f)

    print(f"{args.start_index} to {args.end_index}.")
    if args.start_index != 0 or args.end_index != -1:
        end_index = args.end_index if args.end_index != -1 else len(all_data)
        all_data = all_data[args.start_index:end_index]

    total_price_usd = 0.0  
    for data in tqdm(all_data):
        dataset_name = data['dataset_name']
        video_uid = data['video_uid']
        file_name = data['file_name']

        info_path = f'{args.save_dir}/infos/{dataset_name}/{video_uid}/{file_name}.json'
        #if os.path.exists(info_path):
        #    with open(info_path, 'r') as f:
        #        existing_data = json.load(f)
        #    if 'start_sec' in existing_data:
        #        continue

        action_desc = data['action_description']
        manipulated_object = data['manipulated_object']
        rigid = data['rigid']

        if not rigid:
            continue

        sampling_rate = 1 / camera_cfg.fps
        timestamp = data['timestamp']
        duration = np.round(
            np.arange(
                timestamp-camera_cfg.time_window, 
                timestamp+camera_cfg.time_window, 
                sampling_rate
            ), 
            3
        )
        duration = duration[np.round(np.arange(0, len(duration), len(duration)//8))]

        try:
            clip = []
            for i, _t in enumerate(duration):
                image_path = f'{args.save_dir}/images/{dataset_name}/{video_uid}/{file_name}/{round(_t, 3)}.jpg'
                pil_image = Image.open(image_path)
                draw = ImageDraw.Draw(pil_image)
                text = str(i)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                position = ((pil_image.width - text_width) // 2, pil_image.height - text_height - 100)
                draw.text(position, text, font=font, fill="white")
                clip.append(pil_image)
        except FileNotFoundError:
            continue
        
        try:
            output = gpt4(
                query=action_desc,
                active_object=manipulated_object,
                pil_frames=clip
            )
            total_price_usd += output['total_price']
        except Exception as e:
            print(f"Error: {e}", flush=True)
            continue

        timestamps = re.findall(r'\d+', output['response'])
        timestamps = [int(t) for t in timestamps]

        if len(timestamps) == 2:
            try:
                start_sec = duration[timestamps[0]]
                end_sec = duration[timestamps[1]]
            except IndexError as e:
                print(f"Index error: {e}", flush=True)
                continue
        else:
            if output['response'] == "invalid":
                start_sec, end_sec = None, None
            else:
                print(f"unexpected responce: {output['response']}", flush=True)
                continue

        if args.visualize:
            #  create video
            video = ImageSequenceClip([np.array(img) for img in clip], fps=8)
            video.write_videofile('temp.mp4')

            # create collage
            resize_size = (704, 704)
            num_columns = (len(clip) + 1) // 2
            dst_width = resize_size[0] * num_columns
            dst_height = resize_size[1] * 2 + 500
            dst = Image.new('RGB', (dst_width, dst_height))
            for idx, image in enumerate(clip):
                image = image.resize(resize_size)
                row = idx // num_columns
                col = idx % num_columns
                dst.paste(image, (resize_size[0] * col, resize_size[1] * row))

            draw = ImageDraw.Draw(dst)
            text = f"{action_desc}\n{output['response']}"
            lines = text.split('\n')
            max_line_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
            total_text_height = sum(draw.textbbox((0, 0), line, font=font)[3] for line in lines)
            text_x = (dst_width - max_line_width) // 2
            text_y = resize_size[1] * 2 + (500 - total_text_height) // 2
            draw.text((text_x, text_y), text, font=font, align='center', fill=(255, 255, 255))
            dst.save('temp.jpg')
            import pdb; pdb.set_trace()
        else:
            data['start_sec'] = start_sec
            data['end_sec'] = end_sec
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, 'w') as f:
                json.dump(data, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data dirs
    parser.add_argument("--data_dir", default='./data')
    parser.add_argument("--save_dir", default='/your/path/to/savedir/EgoScaler')
    parser.add_argument("--prompt_path", default='/your/path/to/workdir/EgoScaler/egoscaler/data/prompt/get_timestamp.txt')
    parser.add_argument("--visualize", action='store_true')
    
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)
    args = parser.parse_args()
    
    main(args)

    