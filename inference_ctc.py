import argparse
import torch
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from feature_extractor import OmniASRFeatureExtractor
from omnilingual_ctc import OmnilingualModelForCTC

def main():
    parser = argparse.ArgumentParser(description='Run inference with a Wav2Vec2 CTC model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local Hugging Face model directory.')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file for transcription.')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='Expected audio sampling rate (default: 16000)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    print(f"Loading tokenizer and model from: {args.model_path}")

    # Load tokenizer and feature extractor separately
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    model = OmnilingualModelForCTC.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=model_dtype,
        low_cpu_mem_usage=True
    )

    print(f"Loading audio file: {args.audio_file}")
    audio_array, sr = torchaudio.load(args.audio_file)

    # Resample if necessary
    if sr != args.sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=args.sampling_rate)
        audio_array = resampler(audio_array)

    # Prepare inputs for the model using feature extractor
    inputs = feature_extractor(
        audio_array.squeeze().numpy(),
        sampling_rate=args.sampling_rate,
        return_tensors='pt'
    ).to(model.device, dtype=model_dtype if model_dtype == torch.bfloat16 else torch.float32)

    print("Running inference...")
    with torch.no_grad():
        # Get the logits from the CTC model
        logits = model(
            input_values=inputs['input_values'],
            attention_mask=inputs.get('attention_mask'),
        ).logits

    # Use HuggingFace's built-in CTC decoding to handle repeated characters properly
    predicted_ids = torch.argmax(logits, dim=-1)
    print(tokenizer.pad_token_id, tokenizer.bos_token)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n" + "="*50)
    print(f"Audio file: {args.audio_file}")
    print(f"Transcription: {transcription}")
    print("="*50)

if __name__ == "__main__":
    main()