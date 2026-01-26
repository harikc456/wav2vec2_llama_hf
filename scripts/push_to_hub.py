import argparse
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoTokenizer
from wav2vec2_llama import Wav2Vec2LlamaModel

def main():
    parser = argparse.ArgumentParser(description="Push a converted model to the Hugging Face Hub.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local Hugging Face model directory.')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the repository on the Hugging Face Hub (e.g., "username/model-name").')
    args = parser.parse_args()

    print(f"Loading model and processor from: {args.model_path}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = Wav2Vec2LlamaModel.from_pretrained(args.model_path, device_map='auto')

    print(f"Pushing model and processor to Hub repository: {args.repo_name}")
    model.push_to_hub(args.repo_name)
    processor.push_to_hub(args.repo_name)
    print("Successfully pushed to the Hub.")

if __name__ == "__main__":
    main()
