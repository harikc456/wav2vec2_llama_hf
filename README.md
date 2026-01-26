# Fairseq Omnilingual Speech Recognition Models to Hugging Face Port

This repository provides scripts and code to run inference with pre-converted Fairseq Omnilingual models using the Hugging Face `transformers` library. Supports both:
- **CTC (Connectionist Temporal Classification)** models: Models with feature extractor, wav2vec2 encoder, and final projection layer (without Llama decoder)
- **LLM (Large Language Model)** models: Hybrid Wav2Vec2-LLaMA architectures with both encoder and decoder

## Repository Structure

- `wav2vec2_llama.py`: Model definition for Fairseq2's Omnilingual LLM models (hybrid Wav2Vec2-LLaMA architectures).
- `omnilingual_ctc.py`: Model definition for CTC-based omnilingual speech recognition models.
- `config.py`: Configuration file for the model parameters.
- `inference.py`: Example script to run inference with remote LLM models.
- `inference_ctc.py`: Example script to run inference with remote CTC models.
- `audio/84-121550-0000.flac`: A sample audio file for testing inference.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/harikc456/wav2vec2_llama_hf.git
    cd wav2vec2_llama_hf
    ```

2.  **Install dependencies:**
    Install the required dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Inference with Remote Models

Run inference on audio files using pre-converted models from the Hugging Face Hub.

**For LLM (Large Language Model) models:** (Hybrid Wav2Vec2-LLaMA models)
```bash
# Using a pre-converted model from Hugging Face Hub
python inference.py \
    --model_path harikc456/wav2vec2-llama-300m \
    --audio_file audio/84-121550-0000.flac
```

**For CTC (Connectionist Temporal Classification) models:** (Models without Llama decoder)
```bash
# Using a pre-converted CTC model from Hugging Face Hub
python inference_ctc.py \
    --model_path harikc456/omnilingual-ctc-300m-v2 \
    --audio_file audio/84-121550-0000.flac
```

The scripts will load the remote model, process the audio file, and print the transcribed text. Note that CTC models typically provide more direct ASR-style transcriptions, while LLM models may produce more conversational responses.

## Pre-converted Models

The following models are available on the Hugging Face Hub. You can use them directly for inference without needing to convert them yourself.

- **LLM Model:** `harikc456/wav2vec2-llama-300m`
- **CTC Model:** `harikc456/omnilingual-ctc-300m-v2`

## Acknowledgements

- This work is based on the original Omnilingual models from Fairseq/Meta AI, supporting both CTC and LLM architectures.
- Great appreciation to the Hugging Face team for their work on the `transformers` library and the tools that make sharing and using models straightforward.

## License

This project is licensed under the MIT License.
