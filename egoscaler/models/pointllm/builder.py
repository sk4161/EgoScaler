import os
import logging
from transformers import AutoTokenizer, AutoConfig

from pointllm.utils import disable_torch_init
from pointllm.model import TrajPointLLMForCausalLM
from constant import RT2_TOKEN_TEMPLATE, TIMESTEP_START_TOKEN, TIMESTEP_SEP_TOKEN, TIMESTEP_END_TOKEN


def init_model(args):
    """
    Initialize the model, tokenizer, and point backbone configuration.
    """
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Log the model name
    model_basename = os.path.basename(model_name)
    print(f'[INFO] Model name: {model_basename}')
    logging.warning(f'Model name: {model_basename}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TrajPointLLMForCausalLM(args, config, model_name)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)
    point_backbone_config = model.get_model().point_backbone_config

    return model, tokenizer, point_backbone_config, mm_use_point_start_end


def add_trajectory_token(args, model, tokenizer):
    """
    Add trajectory-related tokens to the tokenizer and resize token embeddings.
    """
    if args.num_bins > 0:
        num_rt2_tokens = args.num_bins
        rt2_tokens = [RT2_TOKEN_TEMPLATE.format(p=p) for p in range(num_rt2_tokens)]
        tokenizer.add_tokens([TIMESTEP_START_TOKEN, TIMESTEP_SEP_TOKEN, TIMESTEP_END_TOKEN])
        tokenizer.add_tokens(rt2_tokens)

    # Resize token embeddings without mean resizing for efficiency
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    return model, tokenizer


def build_model(args):
    """
    Build the model and tokenizer with the necessary configurations.
    """
    model, tokenizer, point_backbone_config, mm_use_point_start_end = init_model(args)
    model, tokenizer = add_trajectory_token(args, model, tokenizer)

    return model, tokenizer, point_backbone_config, mm_use_point_start_end