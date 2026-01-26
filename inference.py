import argparse
import torch
import torchaudio
from wav2vec2_llama import Wav2Vec2LlamaModel
from transformers import AutoTokenizer
from feature_extractor import OmniASRFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Run inference with a Wav2Vec2-LLaMA model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local Hugging Face model directory.')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file for transcription.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    print(f"Loading tokenizer and model from: {args.model_path}")

    # Load tokenizer and feature extractor separately
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    feature_extractor = OmniASRFeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True
    )

    model = Wav2Vec2LlamaModel.from_pretrained(args.model_path, device_map=device, dtype=model_dtype)

    print(f"Loading audio file: {args.audio_file}")
    audio_array, sr = torchaudio.load(args.audio_file)

    # Resample if necessary
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_array = resampler(audio_array)

    # Preprocess audio using feature extractor
    inputs = feature_extractor(
        audio_array.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors='pt'
    ).to(model.device, dtype=model.dtype)

    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            audio_values=inputs['input_values'],
            # audio_attention_mask=inputs['attention_mask'],
        )

    # Decode using tokenizer instead of processor
    transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n" + "="*20)
    print(f"Transcription: {transcription}")
    print("="*20)

if __name__ == "__main__":
    main()
