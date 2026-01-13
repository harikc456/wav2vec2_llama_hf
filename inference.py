import argparse
import torch
import torchaudio
from wav2vec2_llama import Wav2Vec2LlamaModel
from transformers import Wav2Vec2Processor

def main():
    parser = argparse.ArgumentParser(description='Run inference with a Wav2Vec2-LLaMA model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local Hugging Face model directory.')
    parser.add_argument('--audio_file', type=str, required=True, help='Path to the audio file for transcription.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    print(f"Loading processor and model from: {args.model_path}")
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2LlamaModel.from_pretrained(args.model_path, device_map=device, torch_dtype=model_dtype)

    print(f"Loading audio file: {args.audio_file}")
    audio_array, sr = torchaudio.load(args.audio_file)

    # Resample if necessary
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_array = resampler(audio_array)

    inputs = processor(
        audio_array.squeeze(0), 
        sampling_rate=16000, 
        return_tensors='pt'
    ).to(model.device, dtype=model.dtype)

    print("Running inference...")
    with torch.no_grad():
        generated_ids = model.generate(
            audio_values=inputs['input_values'],
            audio_attention_mask=inputs['attention_mask'],
            max_new_tokens=64,
        )
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n" + "="*20)
    print(f"Transcription: {transcription}")
    print("="*20)

if __name__ == "__main__":
    main()
