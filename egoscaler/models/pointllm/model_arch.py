import torch
import torch.nn as nn
from pointllm.model.pointllm import PointLLMLlamaModel, PointLLMLlamaForCausalLM
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast


class TrajPointLLMForCausalLM(PointLLMLlamaForCausalLM):
    """
    A specialized model for trajectory-based causal language modeling.
    Extends PointLLMLlamaForCausalLM with additional functionality.
    """
    def __init__(self, args, config, model_name: str):
        super().__init__(config)
        self.args = args
        self.config = config
        self.model_name = model_name

        # Load pretrained weights
        self.load_pretrained_weights()

        # Freeze or unfreeze specific parts of the model based on arguments
        self._configure_trainable_parameters()

    def load_pretrained_weights(self):
        """
        Load pretrained weights from the parent model.
        """
        parent_model = PointLLMLlamaForCausalLM.from_pretrained(self.model_name, config=self.config)
        parent_state_dict = parent_model.state_dict()
        self.load_state_dict(parent_state_dict, strict=False)

    def _configure_trainable_parameters(self):
        """
        Configure trainable parameters based on the provided arguments.
        """
        if not self.args.unfreeze_pc_encoder:
            for param in self.model.point_backbone.parameters():
                param.requires_grad = False

        if not self.args.unfreeze_language_model:
            for param in self.model.layers.parameters():
                param.requires_grad = False
        else:
            for param in self.model.layers.parameters():
                param.requires_grad = True

        for param in self.model.embed_tokens.parameters():
            param.requires_grad = True
        for param in self.model.rotary_emb.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for the model.
        """
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            return_dict=return_dict
        )
        return outputs

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.FloatTensor] = None,
        max_length: Optional[int] = 20,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: Optional[float] = 1.0,
        do_sample: Optional[bool] = True,
        num_return_sequences: Optional[int] = 1,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate sequences using the model.
        """
        return super().generate(
            input_ids=input_ids,
            point_clouds=point_clouds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=num_return_sequences,
            **kwargs
        )

    def train(self, mode: bool = True):
        """
        Override the train method to selectively freeze/unfreeze parts of the model.
        """
        super(TrajPointLLMForCausalLM, self).train(mode)

        if not self.args.unfreeze_language_model:
            self.model.layers.eval()
            self.model.embed_tokens.train()
            self.model.rotary_emb.train()

        if not self.args.unfreeze_pc_encoder:
            self.model.point_backbone.eval()

        return self