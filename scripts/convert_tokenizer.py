#!/usr/bin/env python
"""
Converts a SentencePiece .model file to Hugging Face tokenizer format.
"""

import os
import argparse
from transformers import LlamaTokenizer


def convert_fairseq2_to_hf(spm_model_path, output_dir):
    """
    Converts a SentencePiece .model file to a Hugging Face tokenizer.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading SentencePiece model from {spm_model_path}...")
    try:
        tokenizer = LlamaTokenizer(vocab_file=spm_model_path)
        
        # Fairseq2 often uses these defaults. Adjust if your model differs:
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })

        print(f"Saving Hugging Face tokenizer to {output_dir}...")
        # This will save both the .model and the converted tokenizer.json (fast version)
        tokenizer.save_pretrained(output_dir)
        
        print("Success! You can now load this via:")
        print(f"AutoTokenizer.from_pretrained('{output_dir}')")
        
    except Exception as e:
        print(f"Error during conversion: {e}")


def main():
    parser = argparse.ArgumentParser(description='Convert a SentencePiece .model file to Hugging Face tokenizer format.')
    parser.add_argument('--spm_model_path', type=str, required=True, help='Path to the SentencePiece .model file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the converted Hugging Face tokenizer.')
    
    args = parser.parse_args()
    
    convert_fairseq2_to_hf(args.spm_model_path, args.output_dir)


if __name__ == "__main__":
    main()