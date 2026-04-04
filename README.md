# R3 Quant: End-to-End Quantization and RL Training Pipeline

This repository provides an automated, end-to-end pipeline to download a base Vision-Language Model (like Qwen2.5-VL), quantize it using GPTQ, apply LoRA, and fine-tune it using Reinforcement Learning (either GRPO or SFT) on the ScienceQA dataset.

## Prerequisites
Ensure you have Python installed. You should also have a compatible GPU to run the quantization and training processes.

First, set up your Python environment and install the required dependencies:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

The entire pipeline is controlled by a single entry point: `main.py`.

### 1. Configuration
Open `main.py` and modify the configuration variables at the top of the file to suit your needs:

```python
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # The Hugging Face repo ID of the base model
RL_METHOD = "GRPO"                             # Choose between "GRPO" and "SFT"
QUANT_BITS = 3                                 # Number of bits for GPTQ quantization
```

### 2. Execute the Pipeline
Run the script:

```bash
python main.py
```

### What happens automatically?
When you run `main.py`, the following steps occur in order:
1. **Environment Setup**: Necessary directories (`data/`, `weights/`, etc.) are created.
2. **Dataset Download**: The ScienceQA validation dataset is downloaded and converted to Parquet format.
3. **Model Download**: The base model specified in `BASE_MODEL_ID` is downloaded to the `weights/` directory.
4. **Quantization**: The base model is quantized to the specified `QUANT_BITS` (e.g., Int3) using GPTQ and saved automatically.
5. **LoRA & RL Training**: The quantized model is loaded, LoRA adapters are applied (freezing the vision encoder), and the model is trained using the specified `RL_METHOD`. The final checkpoints are saved in either `./r3_quant_checkpoints` (for GRPO) or `./sft_baseline_checkpoints` (for SFT).

## Customization
If you wish to modify the specific hyperparameters for training (such as learning rate, batch size, or max steps), you can directly edit the respective trainer class inside the `src/` directory:
- `src/grpo_trainer.py`
- `src/sft_trainer.py`
