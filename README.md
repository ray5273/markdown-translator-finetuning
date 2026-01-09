# OpenWeight Markdown Translator

Fine-tune various open-weight LLMs to translate Korean markdown documents to English while preserving markdown structure.

Supports multiple open-source models including EXAONE, Llama, Qwen, and Gemma.

[한국어 README](./README-kr.md)

## Features

- **Multiple Model Support**: EXAONE 3.5, Llama 3.1, Qwen 2.5, Gemma 2, and other open-weight LLMs
- **Markdown Structure Preservation**: Preserves code blocks, links, tables, and other markdown elements during translation
- **QLoRA Support**: Train on RTX 3090/4080 with 4-bit quantization
- **Flexible Data Generation**: Generate training data via templates, OpenAI, Anthropic, Ollama, or vLLM
- **Comprehensive Evaluation**: BLEU, chrF, COMET + markdown preservation rate metrics

## Requirements

- Python 3.11+
- CUDA 11.8+
- GPU: 12GB+ VRAM (QLoRA), 24GB+ VRAM (LoRA)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
python -m pip install -r requirements.txt
```

## Project Structure

```
├── configs/                 # Configuration files
│   ├── qlora_config.yaml   # QLoRA settings
│   ├── lora_config.yaml    # LoRA settings
│   ├── training_config.yaml # Training settings
│   ├── inference_config.yaml # Inference settings
│   └── hub_config.yaml     # Hugging Face Hub upload settings
├── src/
│   ├── data/               # Data processing
│   ├── model/              # Model loading & Hub upload
│   ├── training/           # Training modules
│   ├── evaluation/         # Evaluation metrics
│   └── inference/          # Inference pipeline
├── scripts/
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── inference.py        # Inference script
│   └── upload_to_hub.py    # Hub upload script
└── data/
    ├── raw/                # Raw data
    └── processed/          # Preprocessed data
```

## Supported Models

| Model | Model ID | VRAM (QLoRA) | Note |
|-------|----------|--------------|------|
| EXAONE 3.5 | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` | 12GB | Korean specialized |
| Llama 3.1 | `meta-llama/Llama-3.1-8B-Instruct` | 10GB | General purpose |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | 10GB | Multilingual |
| Gemma 2 | `google/gemma-2-9b-it` | 12GB | Balanced performance |

## Usage

### 1. Data Preparation

#### Training Data Format

Training data should be in JSONL format:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Korean to English translator..."},
    {"role": "user", "content": "Translate the following...\n\n# Project Introduction"},
    {"role": "assistant", "content": "# Project Introduction"}
  ]
}
```

#### Data Generation Methods

You can generate training data using multiple methods:

##### Method 1: Template-based Generation (Recommended - No API required)

```bash
python scripts/generate_synthetic.py --method template --num-samples 1000
```

- **Pros**: Free, fast, no API key required
- **Output**: `data/synthetic/template_pairs.jsonl`
- Generates diverse Korean-English translation pairs using predefined markdown templates

##### Method 2: OpenAI API-based Generation

```bash
export OPENAI_API_KEY="sk-..."

# Sequential processing (default, slow)
python scripts/generate_synthetic.py --method openai --num-samples 50 --model "gpt-4o"

# Parallel processing (fast! - recommended)
python scripts/generate_synthetic.py --method openai --num-samples 1000 --parallel --max-concurrent 50

# Batch API (50% cost savings!)
python scripts/generate_synthetic.py --method openai --num-samples 1000 --batch --wait
```

- High-quality translation pair generation
- Supports various topics and styles

##### Method 3: Anthropic API-based Generation

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Sequential processing
python scripts/generate_synthetic.py --method anthropic --num-samples 50 --model "claude-sonnet-4-20250514"

# Parallel processing
python scripts/generate_synthetic.py --method anthropic --num-samples 1000 --parallel --max-concurrent 50
```

##### Method 4: Ollama-based Generation (Local LLM - Free!)

```bash
# Install Ollama and download model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve  # Start server (separate terminal)

# Sequential processing
python scripts/generate_synthetic.py --method ollama --model llama3.1:8b --num-samples 100

# Parallel processing (adjust concurrent requests based on GPU memory)
python scripts/generate_synthetic.py --method ollama --model llama3.1:8b --num-samples 100 --parallel --max-concurrent 4

# Custom server URL
python scripts/generate_synthetic.py --method ollama --base-url http://192.168.1.100:11434 --model llama3.1:8b --num-samples 100
```

- **Pros**: Free, privacy-friendly, no internet required
- **Cons**: Requires GPU, slower than API
- **Recommended models**: `llama3.1:8b`, `qwen2.5:7b`, `gemma2:9b`

##### Method 5: vLLM-based Generation (High-performance Local LLM)

```bash
# Install vLLM
pip install vllm

# Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Sequential processing
python scripts/generate_synthetic.py --method vllm --num-samples 100

# Parallel processing (high throughput)
python scripts/generate_synthetic.py --method vllm --num-samples 1000 --parallel --max-concurrent 10

# Custom server URL
python scripts/generate_synthetic.py --method vllm --base-url http://localhost:8000 --num-samples 100
```

- **Pros**: High throughput, OpenAI-compatible API, batch processing optimized
- **Cons**: Complex setup, large GPU recommended
- **Recommended for**: Large-scale data generation

##### Method 6: Sample Data Generation (Testing)

```bash
python scripts/generate_synthetic.py --method sample --num-samples 20
```

- Uses hardcoded samples (demo/testing purposes)

#### High-speed Data Generation (Parallel Processing & Batch API)

Two optimization options for fast large-scale data generation:

##### Async Parallel Processing (`--parallel`)

Uses asyncio to process multiple API requests concurrently.

```bash
# Generate 1000 samples in parallel (max 50 concurrent requests)
python scripts/generate_synthetic.py \
    --method openai \
    --num-samples 1000 \
    --parallel \
    --max-concurrent 50
```

- **Speed**: 10-20x faster than sequential
- **Cost**: Same (regular API pricing)
- **Time for 1000 samples**: ~15-30 minutes

##### OpenAI Batch API (`--batch`)

Uses OpenAI's Batch API for cost savings.

```bash
# Submit batch job (returns immediately)
python scripts/generate_synthetic.py \
    --method openai \
    --num-samples 1000 \
    --batch

# Submit and wait for completion
python scripts/generate_synthetic.py \
    --method openai \
    --num-samples 1000 \
    --batch \
    --wait

# Check batch status
python scripts/generate_synthetic.py \
    --method openai \
    --batch \
    --batch-action status \
    --batch-id batch_xxxxx

# Download results
python scripts/generate_synthetic.py \
    --method openai \
    --batch \
    --batch-action download \
    --batch-id batch_xxxxx
```

- **Cost**: 50% savings!
- **Time**: 1-2 hours (guaranteed within 24 hours)
- **Limitation**: OpenAI only

##### Performance Comparison (1000 samples)

| Method | Time | Cost | Recommended For |
|--------|------|------|-----------------|
| Sequential | ~7 hours | 100% | Small datasets |
| **Parallel** | ~15-30 min | 100% | **Fast results needed** |
| **Batch API** | ~1-2 hours | **50%** | **Cost savings needed** |

#### Data Preprocessing

Preprocess generated data and split into train/valid/test:

```bash
python scripts/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --include-synthetic \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

**Output files**:
- `data/processed/train.jsonl` (80%)
- `data/processed/valid.jsonl` (10%)
- `data/processed/test.jsonl` (10%)

#### Complete Pipeline Example

```bash
# Step 1: Generate synthetic data
python scripts/generate_synthetic.py --method template --num-samples 1000

# Step 2: Preprocess and split
python scripts/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --include-synthetic

# Step 3: Start training
python scripts/train.py --config configs/training_config.yaml --qlora
```

#### Data Generation Method Comparison

| Method | Speed | Cost | Quality | API Required | GPU Required |
|--------|-------|------|---------|--------------|--------------|
| Template | Very Fast | Free | Medium | No | No |
| OpenAI | Medium | Paid | High | Yes | No |
| Anthropic | Medium | Paid | High | Yes | No |
| **Ollama** | Slow | **Free** | High | No | Yes |
| **vLLM** | Fast | **Free** | High | No | Yes |
| Sample | Very Fast | Free | Low | No | No |

#### Local LLM Recommended Models

| Model | VRAM | Quality | Speed | Recommended For |
|-------|------|---------|-------|-----------------|
| `llama3.1:8b` | 8GB | 3/5 | Fast | RTX 3080/4080 |
| `llama3.1:70b` | 40GB+ | 5/5 | Slow | A100/H100 |
| `qwen2.5:7b` | 8GB | 4/5 | Fast | Multilingual |
| `gemma2:9b` | 10GB | 4/5 | Medium | Balanced |

### 2. Training

```bash
# Train with QLoRA (12-16GB VRAM)
python scripts/train.py --config configs/training_config.yaml --qlora

# Train with LoRA (24GB+ VRAM)
python scripts/train.py --config configs/training_config.yaml
```

### 3. Evaluation

```bash
python scripts/evaluate.py --adapter-path outputs/checkpoints/final/adapter
```

### 4. Inference

```bash
# Single file translation
python scripts/inference.py --input docs/README.ko.md --output docs/README.en.md

# Interactive mode
python scripts/inference.py --interactive

# Batch directory translation
python scripts/inference.py --input-dir docs/ko --output-dir docs/en
```

### 5. Upload to Hugging Face Hub

Share your trained model on Hugging Face Hub.

#### Authentication Setup

```bash
# Method 1: CLI login (recommended)
huggingface-cli login

# Method 2: Environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

#### Auto-upload After Training

```bash
# Auto-upload to Hub after training
python scripts/train.py --config configs/training_config.yaml --qlora \
    --push-to-hub \
    --hub-repo-id your-username/markdown-translator

# Upload as private repository
python scripts/train.py --config configs/training_config.yaml --qlora \
    --push-to-hub \
    --hub-repo-id your-org/internal-model \
    --hub-private
```

#### Manual Upload

```bash
# Upload adapter only (smaller size, recommended)
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/markdown-translator

# Upload merged full model (standalone usage)
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/markdown-translator-merged \
    --merge

# Private + custom settings
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/my-translator \
    --config configs/hub_config.yaml \
    --private
```

#### Using Model from Hub

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model (change according to your model)
# Examples: EXAONE, Llama, Qwen, etc.
base_model = AutoModelForCausalLM.from_pretrained(
    "your-base-model-id",  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "your-username/markdown-translator"
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/markdown-translator"
)
```

## Configuration

### QLoRA Settings (`configs/qlora_config.yaml`)

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Settings (`configs/training_config.yaml`)

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| BLEU | N-gram based translation accuracy |
| chrF | Character-level F-score |
| COMET | Neural network based quality evaluation |
| Preservation Rate | Markdown element preservation rate |

## Hardware Requirements

| Method | VRAM | GPU |
|--------|------|-----|
| QLoRA (4-bit) | 12-16GB | RTX 3090, 4080 |
| LoRA (16-bit) | 20-24GB | RTX 4090, A5000 |
| Full Finetuning | 60GB+ | A100 80GB |

## License

MIT License

## References

- [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [Llama 3.1](https://github.com/meta-llama/llama3)
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5)
- [Gemma 2](https://github.com/google-deepmind/gemma)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)
