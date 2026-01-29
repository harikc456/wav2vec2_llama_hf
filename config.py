
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


import torch
import torch.nn.functional as F
from transformers import (
    Wav2Vec2Config,
    LlamaConfig,
    PretrainedConfig,
)

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
        self.lid_marker = vocab_size - 1
        self.context_start = vocab_size - 2
        self.context_end = vocab_size - 3
        self.context_example_start = vocab_size - 4
        self.context_example_end = vocab_size - 5
        self.context_bos = vocab_size - 6
        self.context_eos = vocab_size - 7
        self.regular_segment = vocab_size - 8
        self.last_segment = vocab_size - 9


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
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        # Special token IDs based on fairseq2 implementation (starting from index 9812)
        lid_marker_id: Optional[int] = None,
        context_start_id: Optional[int] = None,
        context_end_id: Optional[int] = None,
        context_example_start_id: Optional[int] = None,
        context_example_end_id: Optional[int] = None,
        context_bos_id: Optional[int] = None,
        context_eos_id: Optional[int] = None,
        regular_segment_id: Optional[int] = None,
        last_segment_id: Optional[int] = None,
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

        # Special token IDs based on fairseq2 implementation
        # These default to the known positions in fairseq2 (starting from 9812)
        self.lid_marker_id = lid_marker_id if lid_marker_id is not None else 9812
        self.context_start_id = context_start_id if context_start_id is not None else 9813
        self.context_end_id = context_end_id if context_end_id is not None else 9814
        self.context_example_start_id = context_example_start_id if context_example_start_id is not None else 9815
        self.context_example_end_id = context_example_end_id if context_example_end_id is not None else 9816
        self.context_bos_id = context_bos_id if context_bos_id is not None else 9817
        self.context_eos_id = context_eos_id if context_eos_id is not None else 9818
        self.regular_segment_id = regular_segment_id if regular_segment_id is not None else 9819
        self.last_segment_id = last_segment_id if last_segment_id is not None else 9820