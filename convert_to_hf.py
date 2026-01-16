import gc
import torch
import argparse
from config import Wav2Vec2LlamaConfig, LlamaConfig
from wav2vec2_llama import Wav2Vec2LlamaModel
from transformers import Wav2Vec2Config

def _map_fairseq2_to_hf_keys(fairseq2_state_dict: dict) -> dict:
    """
    Map fairseq2 state dict keys to HuggingFace format.
    
    This function handles the key naming differences between fairseq2 and HF.
    """
    mapped_dict = {}
    
    for old_key, value in fairseq2_state_dict.items():
        new_key = old_key
        
        # Handle encoder_frontend prefix - map to HF Wav2Vec2 structure
        if old_key.startswith('encoder_frontend.feature_extractor.layers.'):
            # Map: encoder_frontend.feature_extractor.layers.X -> feature_extractor.conv_layers.X
            new_key = old_key.replace('encoder_frontend.feature_extractor.layers.', 
                                     'wav2vec2.feature_extractor.conv_layers.')
        
        elif old_key.startswith('encoder_frontend.post_extract_layer_norm.'):
            # Map: encoder_frontend.post_extract_layer_norm -> feature_projection.layer_norm
            new_key = old_key.replace('encoder_frontend.post_extract_layer_norm.', 
                                     'wav2vec2.feature_projection.layer_norm.')
        
        elif old_key.startswith('encoder_frontend.model_dim_proj.'):
            # Map: encoder_frontend.model_dim_proj -> feature_projection.projection
            new_key = old_key.replace('encoder_frontend.model_dim_proj.', 
                                     'wav2vec2.feature_projection.projection.')
        
        elif old_key.startswith('encoder_frontend.pos_encoder.conv.'):
            # Map: encoder_frontend.pos_encoder.conv -> encoder.pos_conv_embed.conv
            new_key = old_key.replace('encoder_frontend.pos_encoder.conv.', 
                                     'wav2vec2.encoder.pos_conv_embed.conv.')
        
        elif old_key.startswith('encoder_frontend.layer_norm.'):
            # Map: encoder_frontend.layer_norm -> encoder.layer_norm
            new_key = old_key.replace('encoder_frontend.layer_norm.', 
                                     'wav2vec2.encoder.layer_norm.')
        
        elif old_key.startswith('encoder_frontend.layers.'):
            # Map: encoder_frontend.layers.X -> encoder.layers.X
            # Handle layer-specific mappings
            new_key = old_key.replace('encoder_frontend.layers.', 'wav2vec2.encoder.layers.')
            
        
        # Map standalone encoder keys (if not under encoder_frontend)
        elif old_key.startswith('encoder.'):
            temp_key = "wav2vec2." + old_key

            if '.self_attn_layer_norm.' in temp_key:
                new_key = temp_key.replace('.self_attn_layer_norm.', '.layer_norm.')

            elif "self_attn.q_proj" in temp_key:
                new_key = temp_key.replace("self_attn.q_proj", "attention.q_proj")

            elif "self_attn.k_proj" in temp_key:
                new_key = temp_key.replace("self_attn.k_proj", "attention.k_proj")

            elif "self_attn.v_proj" in temp_key:
                new_key = temp_key.replace("self_attn.v_proj", "attention.v_proj")

            elif "self_attn.output_proj" in temp_key:
                new_key = temp_key.replace("self_attn.output_proj", "attention.out_proj")

            elif "ffn.inner_proj" in temp_key:
                new_key = temp_key.replace("ffn.inner_proj", "feed_forward.intermediate_dense")

            elif "ffn.output_proj" in temp_key:
                new_key = temp_key.replace("ffn.output_proj", "feed_forward.output_dense")

            elif "ffn_layer_norm" in temp_key:
                new_key = temp_key.replace("ffn_layer_norm", "final_layer_norm")

            elif "encoder.layer_norm" in temp_key:
                new_key = old_key.replace("encoder.layer_norm", "encoder_layer_norm")

            elif 'pos_encoder.conv.' in temp_key:
                new_key = temp_key.replace('pos_encoder.conv.', 'pos_conv_embed.conv.')
                
                if 'weight_g' in new_key:
                    new_key = new_key.replace('weight_g', 'parametrizations.weight.original0')
                    print(temp_key, new_key, fairseq2_state_dict[old_key].size())
                
                elif 'weight_v' in new_key:
                    new_key = new_key.replace('weight_v', 'parametrizations.weight.original1')
                    print(temp_key, new_key, fairseq2_state_dict[old_key].size())
        
        elif old_key == 'text_frontend.weight':
            new_key = 'llama.model.embed_tokens.weight'
        
        elif old_key.startswith('llama_decoder.'):
            new_key = old_key.replace('llama_decoder.', "llama.model.")
            
            if "self_attn_layer_norm" in new_key:
                new_key = new_key.replace("self_attn_layer_norm", "input_layernorm")

            elif "ffn.inner_proj" in new_key:
                new_key = new_key.replace("ffn.inner_proj", "mlp.up_proj")

            elif "ffn.output_proj" in new_key:
                new_key = new_key.replace("ffn.output_proj", "mlp.down_proj")

            elif "ffn.gate_proj" in new_key:
                new_key = new_key.replace("ffn.gate_proj", "mlp.gate_proj")

            elif "self_attn.output_proj" in new_key:
                new_key = new_key.replace("self_attn.output_proj", "self_attn.o_proj")

            elif "ffn_layer_norm" in new_key:
                new_key = new_key.replace("ffn_layer_norm", "post_attention_layernorm")

            elif ".layer_norm.weight" in new_key:
                new_key = new_key.replace(".layer_norm.weight", ".norm.weight")

        
        # Final projection (LM head)
        elif old_key == 'final_proj.weight':
            new_key = 'llama.lm_head.weight'
        
        mapped_dict[new_key] = value
    
    return mapped_dict

def load_from_fairseq2_checkpoint(
    fairseq2_checkpoint_path: str,
    config: Wav2Vec2LlamaConfig = None,
    output_path: str = None,
    auto_detect_langs: bool = True,
    debug: bool = False
) -> Wav2Vec2LlamaModel:
    """
    Load a pretrained model from fairseq2 checkpoint.
    
    Args:
        fairseq2_checkpoint_path: Path to fairseq2 .pt checkpoint file
        config: Optional config, will be inferred from checkpoint if not provided
        auto_detect_langs: If True, automatically detect number of languages from checkpoint
        debug: If True, print detailed information about key mapping
    
    Returns:
        Loaded Wav2Vec2LlamaModel
    
    Example:
        >>> model = load_from_fairseq2_checkpoint("path/to/checkpoint.pt")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load fairseq2 checkpoint
    print(f"Loading checkpoint from {fairseq2_checkpoint_path}...")
    checkpoint = torch.load(fairseq2_checkpoint_path, map_location='cpu')
    
    # Extract state dict
    state_dict = checkpoint.get('model', checkpoint)
    
    print(f"\nAuto-detecting configuration from checkpoint...")
    
    # 1. Detect vocab size from lm_head (final_proj) - this is the TRUE vocab size
    if 'final_proj.weight' in state_dict:
        detected_vocab_size = state_dict['final_proj.weight'].shape[0]
        print(f"  ✓ Vocab size (from lm_head): {detected_vocab_size}")
        config.vocab_size = detected_vocab_size
    
    # 2. Detect number of special tokens from embedding size
    if 'text_frontend.weight' in state_dict:
        detected_embed_size = state_dict['text_frontend.weight'].shape[0]
        detected_n_special = detected_embed_size - config.vocab_size
        print(f"  ✓ Embedding size: {detected_embed_size}")
        print(f"  ✓ Special tokens: {detected_n_special} (calculated as {detected_embed_size} - {config.vocab_size})")
        config.n_special_tokens = detected_n_special
    
    # 3. Detect encoder hidden size
    if 'encoder.layer_norm.weight' in state_dict:
        detected_encoder_dim = state_dict['encoder.layer_norm.weight'].shape[0]
        print(f"  ✓ Encoder hidden size: {detected_encoder_dim}")
        config.encoder_hidden_size = detected_encoder_dim
    
    # 4. Detect decoder hidden size
    if 'encoder_proj.weight' in state_dict:
        detected_decoder_dim = state_dict['encoder_proj.weight'].shape[0]
        print(f"  ✓ Decoder hidden size: {detected_decoder_dim}")
        config.decoder_hidden_size = detected_decoder_dim
    
    # 5. Detect number of languagespython convert_to_hf.py path/to/checkpoint.pt
    if auto_detect_langs and 'lang_embeddings.weight' in state_dict:
        num_langs = state_dict['lang_embeddings.weight'].shape[0]
        print(f"  ✓ Number of languages: {num_langs}")
        config.num_languages = num_langs
    
    print(f"\nCreating model with detected config...")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Special tokens: {config.n_special_tokens}")
    print(f"  - Total embedding size: {config.vocab_size + config.n_special_tokens}")
    print(f"  - LM head output size: {config.vocab_size}")
    
    # Create model with corrected config
    model = Wav2Vec2LlamaModel(config).to(device)
    
    # Map fairseq2 state dict keys to HuggingFace format
    print(f"\nMapping state dict keys...")
    print(f"\nCreating model on {device}...")
    mapped_state_dict = _map_fairseq2_to_hf_keys(state_dict)
    
    # Load weights
    print(f"\nLoading weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=True)
    
    print(f"\n✓ Model loaded successfully!")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Total parameters: {total_params:.1f}M")

    # 6. Cleanup GPU and RAM memory
    del state_dict
    del mapped_state_dict
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saving converted model to: {output_path}")
    model.save_pretrained(output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a Fairseq Omnilingual model to Hugging Face format.')
    parser.add_argument('--fairseq_checkpoint_path', type=str, required=True, help='Path to the original Fairseq .pt checkpoint file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the converted Hugging Face model.')
    args = parser.parse_args()

    vocab_size = 10288
    n_special_characters = 1

    llama_config = LlamaConfig(
        model_dim=4096,
        vocab_size=vocab_size,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_size=4096,
        rope_theta=10_000.0,
        max_position_embeddings=8192,
        intermediate_size=2816,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5
    )

    wav2vec2_config = Wav2Vec2Config(
        # Feature encoder
        conv_bias=True,
        feat_extract_norm="layer",

        # Transformer encoder
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",

        # Dropout
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,

        # Layer norm
        layer_norm_eps=1e-5,
        do_stable_layer_norm=False,

        # ASR head
        vocab_size=vocab_size,
        final_dropout=0.0,

        # Misc
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
        gradient_checkpointing=False,
    )

    config = Wav2Vec2LlamaConfig(
        wav2vec2_config=wav2vec2_config.to_dict(),
        llama_config=llama_config.to_dict(),
        model_variant="llm_asr",
        encoder_stacking=1,
        n_lang_embeddings=1694,
        n_special_characters=n_special_characters,
        vocab_size=vocab_size
    )
    
    print(f"Loading model from: {args.fairseq_checkpoint_path}")
    load_from_fairseq2_checkpoint(args.fairseq_checkpoint_path, config, args.output_path)
