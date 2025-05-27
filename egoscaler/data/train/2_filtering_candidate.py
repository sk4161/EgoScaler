import json
import argparse
import os
from copy import deepcopy
from distutils.util import strtobool
from typing import List
from glob import glob
from tqdm import tqdm
from llama import Dialog, Llama

class ChatCompletion:
    def __init__(self, args):
        self.args = args
        self.dialog_template = self._load_prompt(args.prompt_path)
        self.generator = self._initialize_generator(args)

    def _load_prompt(self, prompt_path: str) -> List[Dialog]:
        with open(prompt_path, 'r') as f:
            prompt = json.load(f)
        if not isinstance(prompt, list):
            raise ValueError("Prompt JSON must be a list of messages.")
        return prompt

    def _initialize_generator(self, args) -> Llama:
        return Llama.build(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
        )

    def create_dialogs(self, texts: List[str]) -> List[List[Dialog]]:
        dialogs_batch = []
        for text in texts:
            dialogs = deepcopy(self.dialog_template)
            dialogs.append({"role": "user", "content": text})
            dialogs_batch.append(dialogs)
        return dialogs_batch

    def completions(self, texts: List[str]) -> List[str]:
        dialogs_batch = self.create_dialogs(texts)
        results = self.generator.chat_completion(
            dialogs_batch,
            max_gen_len=self.args.max_gen_len,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
        )
        return [result['generation']['content'].strip() for result in results]


def chunkify(lst, chunk_size):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def main(args):
    chatbot = ChatCompletion(args)

    rule_base = [
        "walk", "run", "sit",
        "watch", "look", "read",
        "listen", "talk"
    ]

    # Load data
    all_cands_file = glob(f"{args.data_dir}/cands/*/*/*.json")  # raw candidates
    all_fil_cands_file = glob(f"{args.data_dir}/fil_cands/*/*/*.json")  # already filtered candidates

    all_data = []
    for file_name in tqdm(all_cands_file):
        if file_name.replace('cands', 'fil_cands') in all_fil_cands_file:
            continue
        with open(file_name, 'r') as f:
            data = json.load(f)
        all_data.append(data)

    filtered_data = []
    batch_size = args.batch_size
    total = len(all_data)

    for batch in tqdm(chunkify(all_data, batch_size), desc="Processing data", total=(total + batch_size - 1) // batch_size):
        descriptions = [item['action_description'] for item in batch]

        # Rule-based filtering
        rule_filtered_indices = [
            i for i, desc in enumerate(descriptions)
            if any(rule in desc for rule in rule_base)
        ]

        # Remove entries that matched the rule-based filter
        for idx in sorted(rule_filtered_indices, reverse=True):
            del batch[idx]

        if not batch:
            continue

        # LLM-based filtering
        outputs = chatbot.completions([item['action_description'] for item in batch])

        for data, output in zip(batch, outputs):
            try:
                output_bool = bool(strtobool(output))
            except ValueError:
                print(f"Warning: invalid output '{output}' obtained from '{data['action_description']}'. Skipping.")
                continue

            if output_bool:
                file_name = f"{args.data_dir}/fil_cands/{data['dataset_name']}/{data['video_uid']}/{data['file_name']}.json"
                os.makedirs(os.path.dirname(file_name), exist_ok=True)
                with open(file_name, 'w') as f:
                    json.dump(data, f)

                filtered_data.append(data)

    print(f"Total candidates: {len(filtered_data)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter candidates using rule-based and LLM-based methods.")

    # LLaMA settings
    parser.add_argument('--ckpt_dir', default='/your/path/to/llama3/Meta-Llama-3-70B-Instruct', help='Checkpoint directory for LLaMA.')
    parser.add_argument('--tokenizer_path', default='/your/path/to/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model', help='Path to the tokenizer model.')
    parser.add_argument('--prompt_path', default='/your/path/to/workdir/EgoScaler/egoscaler/data/prompt/filtering_candidate.json', help='Path to the prompt JSON file.')
    parser.add_argument('--max_seq_len', type=int, default=300, help='Maximum sequence length.')
    parser.add_argument('--max_gen_len', type=int, default=100, help='Maximum generation length.')
    parser.add_argument('--max_batch_size', type=int, default=128, help='Maximum batch size for the generator.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for nucleus sampling.')

    parser.add_argument("--data_dir", default='/your/path/to/savedir/EgoScaler', help='Directory containing the input data.')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for processing.')

    args = parser.parse_args()
    main(args)
