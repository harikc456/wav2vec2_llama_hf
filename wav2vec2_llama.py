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
from audio_encoder import Wav2Vec2EncoderStableLayerNorm
from config import Wav2Vec2LlamaConfig

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
            config.llama_config.vocab_size + config.n_special_tokens,
            config.llama_config.model_dim
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
                encoded = F.pad(encoded, (0, 0, 0, n_padding), value=0)
                T = encoded.shape[1]

            encoded = encoded.view(
                B,
                T // self.config.encoder_stacking,
                D * self.config.encoder_stacking
            )

        # Compute sequence lengths
        if attention_mask is not None:
            original_lens = attention_mask.sum(dim=1, dtype=torch.int32)
            seq_lens = torch.div(
                original_lens + self.config.encoder_stacking - 1,
                self.config.encoder_stacking,
                rounding_mode='trunc'
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
        audio_seqs: Optional[torch.Tensor] = None,
        text_seqs: Optional[torch.Tensor] = None,
        lang_seqs: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Embed all modalities separately without using ModalityInput objects.
        Returns embedded sequences for each modality.

        Args:
            audio_seqs: Audio input tensor
            text_seqs: Text input tensor
            lang_seqs: Language ID input tensor
            audio_attention_mask: Attention mask for audio
            dtype: Data type for embeddings

        Returns:
            Tuple of (embedded audio, embedded text, embedded language)
        """
        embedded_audio_seqs = None
        embedded_text_seqs = None
        embedded_lang_seqs = None

        # Handle audio embedding
        if audio_seqs is not None:
            embedded_audio_seqs, _ = self.embed_audio(audio_seqs, attention_mask=audio_attention_mask)

        # Handle text embedding
        if text_seqs is not None:
            embedded_text_seqs = self.embed_text(text_seqs).to(dtype)

        # Handle language embedding
        if lang_seqs is not None:
            embedded_lang_seqs = self.embed_lang(lang_seqs).to(dtype)

        return embedded_audio_seqs, embedded_text_seqs, embedded_lang_seqs
    
    def _concat_inputs(
        self,
        embedded_audio_seqs: Optional[torch.Tensor] = None,
        audio_loss_flags: Optional[List[bool]] = None,
        embedded_text_seqs: Optional[torch.Tensor] = None,
        text_loss_flags: Optional[List[bool]] = None,
        embedded_lang_seqs: Optional[torch.Tensor] = None,
        lang_loss_flags: Optional[List[bool]] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        add_padding: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate inputs without using ModalityInput objects.

        CRITICAL FIX: Loss mask marks positions where we predict the NEXT token.
        Position i in loss_mask=True means we use logits[i] to predict targets[i].

        Returns:
            Tuple of (concatenated embeddings, loss mask)
        """
        # Determine batch size from the first available tensor
        B = None
        hidden_size = None

        if embedded_audio_seqs is not None:
            B = embedded_audio_seqs.shape[0]
            hidden_size = embedded_audio_seqs.shape[-1]
        elif embedded_text_seqs is not None:
            B = embedded_text_seqs.shape[0]
            hidden_size = embedded_text_seqs.shape[-1]
        elif embedded_lang_seqs is not None:
            B = embedded_lang_seqs.shape[0]
            hidden_size = embedded_lang_seqs.shape[-1]

        if B is None or hidden_size is None:
            raise ValueError("At least one modality must be provided")

        # Prepare sequence data for concatenation
        seq_info = []

        if embedded_audio_seqs is not None:
            seq_info.append((embedded_audio_seqs, [embedded_audio_seqs.shape[1]] * B, audio_loss_flags))

        if embedded_text_seqs is not None:
            seq_info.append((embedded_text_seqs, [embedded_text_seqs.shape[1]] * B, text_loss_flags))

        if embedded_lang_seqs is not None:
            seq_info.append((embedded_lang_seqs, [embedded_lang_seqs.shape[1]] * B, lang_loss_flags))

        # Compute total lengths
        total_lens = []
        for b in range(B):
            total_len = sum(seq_lens[b] for _, seq_lens, _ in seq_info)
            total_lens.append(total_len)

        max_len = max(total_lens) if add_padding else max(total_lens)

        # Initialize tensors
        concat_embeds = torch.zeros(B, max_len, hidden_size, device=device, dtype=dtype)
        loss_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        # Fill concatenated sequence
        for b in range(B):
            pos = 0
            for seqs, seq_lens, loss_flags in seq_info:
                length = seq_lens[b]
                if length > 0:
                    concat_embeds[b, pos:pos+length] = seqs[b, :length]

                    # CRITICAL: Loss mask at position i predicts token at position i
                    # We mark from (pos-1) to (pos-1+length) because:
                    # - The model sees tokens at positions [0...pos-1]
                    # - And predicts tokens at positions [pos...pos+length-1]
                    # - So logits[pos-1] predicts the first token of this segment
                    if loss_flags and loss_flags[b] and pos > 0:
                        loss_mask[b, pos-1:pos-1+length] = True

                    pos += length

        return concat_embeds, loss_mask
    
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

        # Embed audio
        embedded_audio, embedded_audio_seq_lens = self.embed_audio(audio_values, attention_mask=audio_attention_mask)
        audio_loss_flags = [False] * B  # Audio segments don't contribute to loss

        # Prepare text and language inputs
        text_loss_flags = [False] * B  # Initialize with default value

        # Add language ID if configured
        if self.lang_embeddings is not None:
            # LID marker
            lid_marker = torch.full((B, 1), self.config.lid_marker_id,
                                   dtype=torch.long, device=device)
            lid_marker_embedded = self.embed_text(lid_marker).to(dtype)

            # Language ID
            if lang_ids is not None:
                lang_id_tensor = lang_ids.unsqueeze(1) if lang_ids.dim() == 1 else lang_ids
            else:
                lang_id_tensor = torch.zeros(B, 1, dtype=torch.long, device=device)

            embedded_lang = self.embed_lang(lang_id_tensor).to(dtype)

            # Combine with language embedding if present
            # Concatenate LID marker, language ID, and BOS
            bos = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            bos_embedded = self.embed_text(bos).to(dtype)
            combined_text = torch.cat([lid_marker_embedded, embedded_lang, bos_embedded], dim=1)
        else:
            # Just BOS token
            bos = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            combined_text = self.embed_text(bos).to(dtype)

        # Add text if provided (training mode)
        if input_ids is not None:
            # Embed the input text
            input_text_embedded = self.embed_text(input_ids).to(dtype)

            # Concatenate with existing text (LID marker + lang ID + BOS)
            combined_text = torch.cat([combined_text, input_text_embedded], dim=1)

            # Update loss flags - Input text contributes to loss
            text_loss_flags = [True] * B  # Input text contributes to loss

            # Add EOS token
            eos = torch.full((B, 1), self.config.eos_token_id, dtype=torch.long, device=device)
            eos_embedded = self.embed_text(eos).to(dtype)
            combined_text = torch.cat([combined_text, eos_embedded], dim=1)

        # Now concatenate all embedded inputs
        inputs_embeds, loss_mask = self._concat_inputs(
            embedded_audio_seqs=embedded_audio,
            audio_loss_flags=audio_loss_flags,
            embedded_text_seqs=combined_text,
            text_loss_flags=text_loss_flags,
            embedded_lang_seqs=None,  # Already handled above
            lang_loss_flags=None,
            device=device,
            dtype=dtype,
            add_padding=(labels is not None)
        )

        # Calculate total lens from the concatenated inputs
        total_lens = [inputs_embeds.shape[1]] * B

        # Create attention mask using torch.arange for efficiency
        max_len = inputs_embeds.shape[1]
        position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        inputs_attention_mask = (position_ids < torch.tensor(total_lens, device=device).unsqueeze(1)).long()

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
        """
        B = audio_values.shape[0]
        device = audio_values.device
        dtype = next(self.parameters()).dtype

        # Embed audio
        embedded_audio, embedded_audio_seq_lens = self.embed_audio(audio_values, attention_mask=audio_attention_mask)

        # Prepare text and language inputs
        text_loss_flags = [False] * B  # Initialize with default value

        # Add language embeddings if configured
        if self.lang_embeddings is not None:
            # LID marker
            lid_marker = torch.full((B, 1), self.config.lid_marker_id,
                                   dtype=torch.long, device=device)
            lid_marker_embedded = self.embed_text(lid_marker).to(dtype)

            # Language ID
            if lang_ids is not None:
                lang_id_tensor = lang_ids.unsqueeze(1) if lang_ids.dim() == 1 else lang_ids
            else:
                lang_id_tensor = torch.zeros(B, 1, dtype=torch.long, device=device)

            embedded_lang = self.embed_lang(lang_id_tensor).to(dtype)

            # Combine with language embedding if present
            # Concatenate LID marker, language ID, and BOS
            bos = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            bos_embedded = self.embed_text(bos).to(dtype)
            combined_text = torch.cat([lid_marker_embedded, embedded_lang, bos_embedded], dim=1)
        else:
            # Just BOS token
            bos = torch.full((B, 1), self.config.bos_token_id, dtype=torch.long, device=device)
            combined_text = self.embed_text(bos).to(dtype)

        # Add empty token for generation (this is where the model will start predicting)
        empty_token = torch.zeros(B, 1, dtype=torch.long, device=device)
        empty_token_embedded = self.embed_text(empty_token).to(dtype)
        combined_text = torch.cat([combined_text, empty_token_embedded], dim=1)

        # Update loss flags - The empty token contributes to loss (will be predicted)
        text_loss_flags = [True] * B  # The empty token contributes to loss (will be predicted)

        # EOS token
        eos = torch.full((B, 1), self.config.eos_token_id, dtype=torch.long, device=device)
        eos_embedded = self.embed_text(eos).to(dtype)
        combined_text = torch.cat([combined_text, eos_embedded], dim=1)

        # Concatenate (no padding for generation)
        inputs_embeds, _ = self._concat_inputs(
            embedded_audio_seqs=embedded_audio,
            audio_loss_flags=[False] * B,  # Audio segments don't contribute to loss
            embedded_text_seqs=combined_text,
            text_loss_flags=text_loss_flags,
            embedded_lang_seqs=None,  # Already handled above
            lang_loss_flags=None,
            device=device,
            dtype=dtype,
            add_padding=False  # No padding for generation
        )

        # Calculate total lens from the concatenated inputs
        total_lens = [inputs_embeds.shape[1]] * B

        # Trim to actual lengths
        max_context_len = max(total_lens) - 1
        inputs_embeds = inputs_embeds[:, :max_context_len, :]

        # Create attention mask using torch.arange for efficiency
        position_ids = torch.arange(max_context_len, device=device).unsqueeze(0).expand(B, -1)
        attention_mask = (position_ids < torch.tensor(total_lens, device=device).unsqueeze(1)).long()

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