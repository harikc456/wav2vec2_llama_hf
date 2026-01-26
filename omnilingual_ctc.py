from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from audio_encoder import Wav2Vec2EncoderStableLayerNorm


class OmnilingualModelForCTC(PreTrainedModel):
    """
    Wav2Vec2 encoder + CTC layer for automatic speech recognition.
    Uses the custom Wav2Vec2 architecture from wav2vec2_llama.py.
    """

    config_class = Wav2Vec2Config

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

        self.config = config

        # Audio encoder (Wav2Vec2 with custom encoder)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.wav2vec2.encoder = Wav2Vec2EncoderStableLayerNorm(config)

        # Layer norm BEFORE projection (matches fairseq2)
        self.encoder_layer_norm = nn.LayerNorm(config.hidden_size)

        # CTC layer for speech recognition (following HuggingFace convention)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature extractor
        so that its parameters will not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing convolution and attention on padding token indices.
            labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # Apply layer norm before projection (matching fairseq2)
        hidden_states = self.encoder_layer_norm(hidden_states)

        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        # Apply LM head to get logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Calculate log probabilities for CTC
            log_probs = F.log_softmax(logits.transpose(0, 1), dim=-1)  # (T, B, V)

            # Calculate input and target lengths
            input_lengths = None
            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=1)
            else:
                input_lengths = torch.full(
                    (input_values.shape[0],), 
                    input_values.shape[-1] // 320,  # Approximate: raw waveform length / 320 (conv subsampling)
                    dtype=torch.long,
                    device=input_values.device
                )
                
            # Apply encoder stacking to input lengths
            if self.config.encoder_stacking > 1:
                input_lengths = torch.ceil(input_lengths.float() / self.config.encoder_stacking).long()

            # Convert labels to int and calculate target lengths
            labels = labels.int()
            target_lengths = (labels != self.config.pad_token_id).sum(dim=1).int()

            # Calculate CTC loss
            loss = F.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=self.config.pad_token_id,
                reduction='mean',
                zero_infinity=True
            )

            # Apply CTC loss weight
            loss = loss * self.config.ctc_loss_weight

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )