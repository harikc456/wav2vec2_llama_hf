"""
Audio encoder components shared between different model variants.
This module contains classes and functions related to audio encoding that are reused across multiple files.
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PositionalConvEmbedding, Wav2Vec2EncoderLayerStableLayerNorm


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    """
    Custom Wav2Vec2 encoder with stable layer normalization.
    This class is shared between different model variants.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)

        synced_gpus = False

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = self.training and dropout_probability < self.config.layerdrop
            if not skip_the_layer or synced_gpus:
                # under fsdp or deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )