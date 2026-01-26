import argparse
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoTokenizer
from omnilingual_ctc import OmnilingualModelForCTC


def main():
    parser = argparse.ArgumentParser(description="Push a converted CTC model to the Hugging Face Hub.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the local Hugging Face CTC model directory.')
    parser.add_argument('--repo_name', type=str, required=True, help='Name of the repository on the Hugging Face Hub (e.g., "username/model-name").')
    args = parser.parse_args()

    print(f"Loading CTC model and processor from: {args.model_path}")
    
    # Load feature extractor and tokenizer

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True
    )
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Create processor from feature extractor and tokenizer
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Load the CTC model
    model = OmnilingualModelForCTC.from_pretrained(args.model_path, device_map='auto')

    print(f"Pushing CTC model and processor to Hub repository: {args.repo_name}")
    
    # Push model and processor to hub
    processor.push_to_hub(args.repo_name)
    model.push_to_hub(args.repo_name)
    
    
    print("Successfully pushed CTC model to the Hub.")


if __name__ == "__main__":
    main()