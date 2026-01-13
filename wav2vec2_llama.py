"""
Wav2Vec2-LLaMA Model for ASR - HuggingFace Port
Ports the fairseq2 implementation to use HuggingFace Transformers components.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Model,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder
from transformers.modeling_outputs import BaseModelOutput

from config import Wav2Vec2LlamaConfig, Modality, ModalityInput, Wav2Vec2LlamaSpecialTokens

class Wav2Vec2EncoderNoLN(Wav2Vec2Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer_norm = None 

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, 
                output_hidden_states=False, return_dict=True):
        
        hidden_states = self.pos_conv_embed(hidden_states)
        hidden_states = self.dropout(hidden_states)
    
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
    
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # FIX: Layer returns tuple, extract hidden_states
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]  # <-- THIS IS THE KEY FIX
            
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
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
        self.wav2vec2.encoder = Wav2Vec2EncoderNoLN(config.wav2vec2_config)
        
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
        self.special_tokens = Wav2Vec2LlamaSpecialTokens(
            config.llama_config.vocab_size
        )
        
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
        add_padding: bool = True
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
        
        # Add SDPA workaround padding only during training
        if add_padding:
            max_len += 1
        
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
        num_beams: int = 5,
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
        
        # CRITICAL: BOS is part of context for generation
        bos = torch.full((B, 1), self.config.bos_token_id, 
                        dtype=torch.long, device=device)
        inputs.append(ModalityInput(
            modality=Modality.TEXT,
            seqs=bos,
            seq_lens=[1] * B,
            loss=False,
            embedded=False
        ))
        
        # Embed all inputs
        inputs = self.embed_inputs(inputs, dtype)
        
        # Concatenate (no padding for generation)
        inputs_embeds, _, total_lens = self._concat_inputs(
            inputs, device, dtype, add_padding=False
        )
        
        # Trim to actual lengths
        max_context_len = max(total_lens)
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