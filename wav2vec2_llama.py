"""
Wav2Vec2-LLaMA Model for ASR - HuggingFace Port
Ports the fairseq2 implementation to use HuggingFace Transformers components.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    LlamaForCausalLM,
    LlamaConfig,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PositionalConvEmbedding, Wav2Vec2EncoderLayerStableLayerNorm
from transformers.modeling_outputs import BaseModelOutput


class ModelType(str, Enum):
    """Model variant types"""
    LLM_ASR = "llm_asr"
    LLM_ASR_LID = "llm_asr_lid"  # With Language ID
    ZERO_SHOT = "zero_shot"  # With in-context learning


class Modality(str, Enum):
    """Input modality types"""
    AUDIO = "audio"
    TEXT = "text"
    LANG = "lang"


@dataclass
class ModalityInput:
    """Container for multi-modal inputs"""
    modality: Modality
    seqs: torch.Tensor
    seq_lens: List[int]
    loss: bool = False
    embedded: bool = False


@dataclass
class Wav2Vec2LlamaSpecialTokens:
    """Special tokens for different syntaxes"""
    def __init__(self, vocab_size: int):
        # Use last tokens in vocab for special markers
        self.lid_marker = vocab_size
        self.context_start = vocab_size
        self.context_end = vocab_size + 1
        self.context_example_start = vocab_size + 2
        self.context_example_end = vocab_size + 3
        self.context_bos = vocab_size + 4
        self.context_eos = vocab_size + 5
        self.regular_segment = vocab_size + 2
        self.last_segment = vocab_size + 1


class Wav2Vec2LlamaConfig(PretrainedConfig):
    """Configuration class for Wav2Vec2-LLaMA model"""
    
    model_type = "wav2vec2_llama"
    
    def __init__(
        self,
        wav2vec2_config: Optional[dict] = None,
        llama_config: Optional[dict] = None,
        model_variant: str = "llm_asr",
        encoder_stacking: int = 1,
        max_generation_length: int = 8192,
        lang_embeddings_p: float = 0.0,
        n_lang_embeddings: Optional[int] = None,
        n_special_tokens: Optional[int] = 0,
        n_context_examples: int = 0,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Initialize sub-configs
        self.wav2vec2_config = Wav2Vec2Config(**wav2vec2_config) if wav2vec2_config else Wav2Vec2Config()
        self.llama_config = LlamaConfig(**llama_config) if llama_config else LlamaConfig()
        
        # Model-specific parameters
        self.model_variant = ModelType(model_variant)
        self.encoder_stacking = encoder_stacking
        self.max_generation_length = max_generation_length
        self.lang_embeddings_p = lang_embeddings_p
        self.n_lang_embeddings = n_lang_embeddings
        self.n_context_examples = n_context_examples
        self.n_special_tokens = n_special_tokens


class Wav2Vec2EncoderStableLayerNorm(nn.Module):
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


class Wav2Vec2LlamaModel(PreTrainedModel):
    """
    Wav2Vec2 encoder + LLaMA decoder for automatic speech recognition.
    Fixed to match fairseq2 implementation behavior.
    """
    
    config_class = Wav2Vec2LlamaConfig
    
    def __init__(self, config: Wav2Vec2LlamaConfig):
        super().__init__(config)
        
        self.config = config
        self.model_type = config.model_variant
        
        # Audio encoder (Wav2Vec2)
        self.wav2vec2 = Wav2Vec2Model(config.wav2vec2_config)
        self.wav2vec2.encoder = Wav2Vec2EncoderStableLayerNorm(config.wav2vec2_config)
        
        # Layer norm BEFORE projection (matches fairseq2)
        self.encoder_layer_norm = nn.LayerNorm(config.wav2vec2_config.hidden_size)
        
        # Projection from encoder to decoder dimension
        # Note: stacking is applied BEFORE projection in fairseq2
        encoder_dim = config.wav2vec2_config.hidden_size * config.encoder_stacking
        decoder_dim = config.llama_config.hidden_size
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        
        # Language embeddings (for LID variant)
        self.lang_embeddings = None
        if config.n_lang_embeddings is not None:
            self.lang_embeddings = nn.Embedding(
                config.n_lang_embeddings,
                config.llama_config.hidden_size
            )
        
        # LLaMA decodern_special_tokens
        self.llama = LlamaForCausalLM(config.llama_config)
        self.llama.model.embed_tokens = nn.Embedding(
            config.llama_config.vocab_size + config.n_special_characters, 
            config.llama_config.model_dim
        )

        # Special tokens
        self.special_tokens = Wav2Vec2LlamaSpecialTokens(9812)  # Hardcoding 9812 for now
        
        self.post_init()
    
    def embed_audio(
        self, 
        audio_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """Encode audio matching fairseq2's embed_audio exactly"""

        outputs = self.wav2vec2(
            audio_values,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        encoded = outputs.last_hidden_state

        # Layer norm BEFORE stacking
        encoded = self.encoder_layer_norm(encoded)
        
        # Stack frames
        B, T, D = encoded.shape
        if self.config.encoder_stacking > 1:
            if T % self.config.encoder_stacking != 0:
                n_padding = self.config.encoder_stacking - (T % self.config.encoder_stacking)
                encoded = F.pad(encoded, (0, 0, 0, n_padding))
                T = encoded.shape[1]
            
            encoded = encoded.view(
                B,
                T // self.config.encoder_stacking,
                D * self.config.encoder_stacking
            )
        
        # Compute sequence lengths
        if attention_mask is not None:
            original_lens = attention_mask.sum(dim=1)
            seq_lens = torch.where(
                (original_lens % self.config.encoder_stacking) == 0,
                original_lens // self.config.encoder_stacking,
                original_lens // self.config.encoder_stacking + 1
            ).tolist()
        else:
            seq_lens = [encoded.shape[1]] * B
        
        # Project AFTER stacking
        encoded = self.encoder_proj(encoded)
        return encoded, seq_lens
    
    def embed_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed text tokens"""
        return self.llama.model.embed_tokens(input_ids)
    
    def embed_lang(self, lang_ids: torch.Tensor) -> torch.Tensor:
        """Embed language IDs"""
        if self.lang_embeddings is None:
            raise ValueError("Language embeddings not initialized")
        return self.lang_embeddings(lang_ids)
    
    def embed_inputs(
        self,
        inputs: List[ModalityInput],
        dtype: torch.dtype
    ) -> List[ModalityInput]:
        """
        Embed all modalities, matching fairseq2's embed_inputs.
        Modifies inputs in-place.
        """
        for inp in inputs:
            if inp.embedded:
                continue
            
            # Handle zero-length sequences
            zero_indices = [i for i, length in enumerate(inp.seq_lens) if length == 0]
            if zero_indices:
                max_len = inp.seqs.size(-1)
                for i in zero_indices:
                    inp.seq_lens[i] = max_len
            
            # Embed based on modality
            if inp.modality == Modality.AUDIO:
                inp.seqs, inp.seq_lens = self.embed_audio(inp.seqs)
            elif inp.modality == Modality.TEXT:
                inp.seqs = self.embed_text(inp.seqs).to(dtype)
            elif inp.modality == Modality.LANG:
                inp.seqs = self.embed_lang(inp.seqs).to(dtype)
            else:
                raise ValueError(f"Unknown modality: {inp.modality}")
            
            inp.embedded = True
            
            # Restore zero lengths
            if zero_indices:
                for i in zero_indices:
                    inp.seq_lens[i] = 0
        
        return inputs
    
    def _concat_inputs(
        self,
        inputs: List[ModalityInput],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Concatenate inputs matching fairseq2's concat_inputs.
        
        CRITICAL FIX: Loss mask marks positions where we predict the NEXT token.
        Position i in loss_mask=True means we use logits[i] to predict targets[i].
        """
        B = inputs[0].seqs.shape[0]
        hidden_size = inputs[0].seqs.shape[-1]
        
        # Compute total lengths
        total_lens = [
            sum(inp.seq_lens[b] for inp in inputs)
            for b in range(B)
        ]
        max_len = max(total_lens)
        
        # Initialize tensors
        concat_embeds = torch.zeros(B, max_len, hidden_size, device=device, dtype=dtype)
        loss_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        
        # Fill concatenated sequence
        for b in range(B):
            pos = 0
            for inp in inputs:
                length = inp.seq_lens[b]
                if length > 0:
                    concat_embeds[b, pos:pos+length] = inp.seqs[b, :length]
                    
                    # CRITICAL: Loss mask at position i predicts token at position i
                    # We mark from (pos-1) to (pos-1+length) because:
                    # - The model sees tokens at positions [0...pos-1]
                    # - And predicts tokens at positions [pos...pos+length-1]
                    # - So logits[pos-1] predicts the first token of this segment
                    if inp.loss and pos > 0:
                        loss_mask[b, pos-1:pos-1+length] = True
                    
                    pos += length
        
        return concat_embeds, loss_mask, total_lens
    
    def forward(
        self,
        audio_values: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass - FIXED to match fairseq2 exactly.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if audio_values is None:
            raise ValueError("audio_values must be provided")
        
        B = audio_values.shape[0]
        device = audio_values.device
        dtype = next(self.parameters()).dtype
        
        # Create un-embedded inputs (fairseq2 approach)
        audio_seq_lens = (audio_attention_mask.sum(dim=1).tolist() 
                         if audio_attention_mask is not None 
                         else [audio_values.shape[1]] * B)
        
        inputs = [ModalityInput(
            modality=Modality.AUDIO,
            seqs=audio_values,
            seq_lens=audio_seq_lens,
            loss=False,
            embedded=False
        )]
        
        # Add language ID if configured
        if self.lang_embeddings is not None:
            # LID marker first
            lid_marker = torch.full((B, 1), self.special_tokens.lid_marker, 
                                   dtype=torch.long, device=device)
            inputs.append(ModalityInput(
                modality=Modality.TEXT,
                seqs=lid_marker,
                seq_lens=[1] * B,
                loss=False,
                embedded=False
            ))
            
            # Language ID second
            if lang_ids is not None:
                lang_id_tensor = lang_ids.unsqueeze(1) if lang_ids.dim() == 1 else lang_ids
            else:
                lang_id_tensor = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            inputs.append(ModalityInput(
                modality=Modality.LANG,
                seqs=lang_id_tensor,
                seq_lens=[1] * B,
                loss=False,
                embedded=False
            ))
        
        # BOS token
        bos = torch.full((B, 1), self.config.bos_token_id, 
                        dtype=torch.long, device=device)
        inputs.append(ModalityInput(
            modality=Modality.TEXT,
            seqs=bos,
            seq_lens=[1] * B,
            loss=False,
            embedded=False
        ))
        
        # Add text if provided (training mode)
        if input_ids is not None:
            text_lens = (input_ids != self.config.pad_token_id).sum(dim=1).tolist()
            inputs.append(ModalityInput(
                modality=Modality.TEXT,
                seqs=input_ids,
                seq_lens=text_lens,
                loss=True,
                embedded=False
            ))
            
            # EOS token
            eos = torch.full((B, 1), self.config.eos_token_id, 
                           dtype=torch.long, device=device)
            
            inputs.append(ModalityInput(
                modality=Modality.TEXT,
                seqs=eos,
                seq_lens=[1] * B,
                loss=True,
                embedded=False
            ))
        
        # Embed all inputs
        inputs = self.embed_inputs(inputs, dtype)
        
        # Concatenate
        inputs_embeds, loss_mask, total_lens = self._concat_inputs(
            inputs, device, dtype, add_padding=(labels is not None)
        )
        
        # Create attention mask
        max_len = inputs_embeds.shape[1]
        inputs_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, length in enumerate(total_lens):
            inputs_attention_mask[i, :length] = 1
        
        # Forward through LLaMA
        outputs = self.llama.model(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.llama.lm_head(hidden_states)
        
        # Crop to true lengths
        max_true_len = max(total_lens)
        logits = logits[:, :max_true_len, :]
        loss_mask = loss_mask[:, :max_true_len]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # CRITICAL FIX: Extract targets correctly
            # Add EOS to labels
            labels_with_eos = torch.cat([
                labels,
                torch.full((B, 1), self.config.pad_token_id, 
                          device=labels.device, dtype=labels.dtype)
            ], dim=1)
            
            # Place EOS at end of each sequence
            label_lens = (labels != self.config.pad_token_id).sum(dim=1)
            for i in range(B):
                labels_with_eos[i, label_lens[i]] = self.config.eos_token_id
            
            # Flatten targets (only where loss_mask is True)
            targets_list = []
            for b in range(B):
                # Count how many True values in loss_mask for this batch
                n_loss_tokens = loss_mask[b].sum().item()
                # Take that many tokens from labels_with_eos
                targets_list.append(labels_with_eos[b, :n_loss_tokens])
            
            targets = torch.cat(targets_list, dim=0)
            
            # Extract relevant logits
            relevant_logits = logits[loss_mask]
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.config.pad_token_id,
                reduction='sum'
            )
            loss = loss_fct(relevant_logits, targets)
            
            # Normalize (fairseq2 style)
            n_tokens = (labels != self.config.pad_token_id).sum() + B
            loss = loss / n_tokens * B
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        audio_values: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate transcription - FIXED to match fairseq2.
        
        Key fixes:
        1. Create syntax THEN embed (fairseq2 order)
        2. Don't add SDPA padding for generation
        3. BOS is part of context, not first generated token
        """
        B = audio_values.shape[0]
        device = audio_values.device
        dtype = next(self.parameters()).dtype
        
        # Compute audio sequence lengths
        audio_seq_lens = (audio_attention_mask.sum(dim=1).tolist() 
                         if audio_attention_mask is not None 
                         else [audio_values.shape[1]] * B)
        
        # Create un-embedded inputs
        inputs = [ModalityInput(
            modality=Modality.AUDIO,
            seqs=audio_values,
            seq_lens=audio_seq_lens,
            loss=False,
            embedded=False
        )]
        
        # Add language embeddings if configured
        if self.lang_embeddings is not None:
            # LID marker
            lid_marker = torch.full((B, 1), self.special_tokens.lid_marker, 
                                   dtype=torch.long, device=device)
            inputs.append(ModalityInput(
                modality=Modality.TEXT,
                seqs=lid_marker,
                seq_lens=[1] * B,
                loss=False,
                embedded=False
            ))
            
            # Language ID
            if lang_ids is not None:
                lang_id_tensor = lang_ids.unsqueeze(1) if lang_ids.dim() == 1 else lang_ids
            else:
                lang_id_tensor = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            inputs.append(ModalityInput(
                modality=Modality.LANG,
                seqs=lang_id_tensor,
                seq_lens=[1] * B,
                loss=False,
                embedded=False
            ))

        # Adding BOS, target seqs and EOS
        bos = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        eos = torch.full((B, 1), self.config.eos_token_id, dtype=torch.long, device=device)

        inputs.append(ModalityInput(
            modality=Modality.TEXT,
            seqs=bos,
            seq_lens=[1] * B,
            loss=False,
            embedded=False
        ))

        inputs.append(ModalityInput(
                modality=Modality.TEXT,
                seqs=torch.zeros(1, 1, dtype=torch.long, device=device),
                seq_lens=[1] * B,
                loss=True,
        ))

        inputs.append(ModalityInput(
            modality=Modality.TEXT,
            seqs=eos,
            seq_lens=[1] * B,
            loss=False,
            embedded=False
        ))

        
        # Embed all inputs
        inputs = self.embed_inputs(inputs, dtype)
        # Concatenate (no padding for generation)
        inputs_embeds, _, total_lens = self._concat_inputs(inputs, device, dtype)
        
        # Trim to actual lengths
        max_context_len = max(total_lens) - 1
        inputs_embeds = inputs_embeds[:, :max_context_len, :]
        
        # Create attention mask
        attention_mask = torch.zeros(B, max_context_len, dtype=torch.long, device=device)
        for i, length in enumerate(total_lens):
            attention_mask[i, :length] = 1

        # Generate using HuggingFace's generate
        outputs = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            bos_token_id=self.config.bos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            **generate_kwargs
        )
        
        return outputs