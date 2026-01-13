# Fairseq Omnilingual Wav2Vec2-LLaMA to Hugging Face Port

This repository provides scripts and code to convert the Fairseq Omnilingual model, a hybrid Wav2Vec2-LLaMA architecture, into a format compatible with the Hugging Face `transformers` library. The goal is to make this powerful speech recognition model more accessible to the wider community by leveraging the Hugging Face ecosystem.

## Project Goal

The primary objective is to create a reliable and easy-to-use pipeline for porting the original Fairseq checkpoints to a `transformers`-native format. This allows users to:
- Run inference using the familiar `AutoModel` and `pipeline` APIs.
- Share the converted model on the Hugging Face Hub.
- Integrate the model into other `transformers`-based projects and applications.

## Repository Structure

- `wav2vec2_llama.py`: The core model definition in Hugging Face `transformers` format.
- `config.py`: Configuration file for the model parameters.
- `convert_to_hf.py`: Script to convert original Fairseq checkpoints to the new Hugging Face format.
- `inference.py`: Example script to run inference with the converted model.
- `push_to_hub.py`: Utility script to upload the converted model to the Hugging Face Hub.
- `84-121550-0000.flac`: A sample audio file for testing inference.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/wav2vec2_llama_hf.git
    cd wav2vec2_llama_hf
    ```

2.  **Install dependencies:**
    Install the required dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The process is broken down into three main steps: converting the model, running inference, and (optionally) pushing it to the Hub.

### 1. Convert Fairseq Checkpoint

First, you need to convert the original Fairseq model checkpoint to the Hugging Face format.

```bash
python convert_to_hf.py \
    --fairseq_checkpoint_path /path/to/your/fairseq/model.pt \
    --output_path ./hf_model
```
This will create a new directory (`./hf_model` in this case) containing the `pytorch_model.bin`, `config.json`, and other necessary files for a Hugging Face model.

### 2. Run Inference

Once the model is converted, you can run inference on an audio file. The repository includes a sample FLAC file for quick testing.

```bash
python inference.py \
    --model_path harikc456/wav2vec2-llama-300m \
    --audio_file 84-121550-0000.flac
```
The script will load the converted model, process the audio file, and print the transcribed text.

### 3. Push to Hugging Face Hub

To share your converted model with the community, you can upload it to the Hugging Face Hub.

First, make sure you are logged in to your Hugging Face account:
```bash
huggingface-cli login
```

Then, run the push script:
```bash
python push_to_hub.py \
    --model_path ./hf_model \
    --repo_name "your-username/your-model-name"
```
This will create a new repository on the Hub under your username and upload the model files.

## Pre-converted Models

The 300m LLM variant of the model is available on the Hugging Face Hub. You can use it directly for inference without needing to convert it yourself.

- **Model:** `harikc456/wav2vec2-llama-300m`
- **Link:** [https://huggingface.co/harikc456/wav2vec2-llama-300m](https://huggingface.co/harikc456/wav2vec2-llama-300m)

To use this model, you can load it directly with `AutoModelForCTC` and `AutoProcessor` from the `transformers` library:

```python
from transformers import AutoModelForCTC, AutoProcessor

model_id = "harikc456/wav2vec2-llama-300m"

model = AutoModelForCTC.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
```

## Acknowledgements

- This work is heavily based on the original Omnilingual model from Fairseq/Meta AI.
- Great appreciation to the Hugging Face team for their work on the `transformers` library and the tools that make sharing and using models straightforward.

## License

This project is licensed under the MIT License.
