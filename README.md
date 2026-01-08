# Markdown Translator Fine-tuning

한국어 마크다운 문서를 영어로 번역하기 위한 LLM 파인튜닝 프레임워크입니다.

## Features

- **모델 선택 자유**: 어떤 HuggingFace 모델이든 파인튜닝 가능
  - LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
  - meta-llama/Llama-3.1-8B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
  - 그 외 모든 CausalLM 모델
- **마크다운 구조 보존**: 코드 블록, 링크, 테이블 등 마크다운 요소를 보존하며 번역
- **QLoRA 지원**: 4-bit 양자화로 RTX 3090/4080에서도 학습 가능
- **다양한 평가 메트릭**: BLEU, chrF, COMET + 마크다운 보존율 평가

## Requirements

- Python 3.11+
- CUDA 11.8+
- GPU: 12GB+ VRAM (QLoRA), 24GB+ VRAM (LoRA)

## Installation

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# 의존성 설치
python -m pip install -r requirements.txt
```

## Project Structure

```
├── configs/                 # 설정 파일
│   ├── qlora_config.yaml   # QLoRA 설정
│   ├── lora_config.yaml    # LoRA 설정
│   ├── training_config.yaml # 학습 설정
│   ├── inference_config.yaml # 추론 설정
│   └── hub_config.yaml     # Hugging Face Hub 업로드 설정
├── src/
│   ├── data/               # 데이터 처리
│   ├── model/              # 모델 로딩 및 Hub 업로드
│   ├── training/           # 학습 모듈
│   ├── evaluation/         # 평가 메트릭
│   └── inference/          # 추론 파이프라인
├── scripts/
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   ├── inference.py        # 추론 스크립트
│   └── upload_to_hub.py    # Hub 업로드 스크립트
└── data/
    ├── raw/                # 원본 데이터
    └── processed/          # 전처리된 데이터
```

## Usage

### 1. 데이터 준비

학습 데이터를 JSONL 형식으로 준비합니다:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Korean to English translator..."},
    {"role": "user", "content": "Translate the following...\n\n# 프로젝트 소개"},
    {"role": "assistant", "content": "# Project Introduction"}
  ]
}
```

### 2. 학습

```bash
# 모델 지정 + QLoRA로 학습 (12-16GB VRAM)
python scripts/train.py --model "meta-llama/Llama-3.1-8B-Instruct" --qlora

# 다른 모델로 학습
python scripts/train.py --model "Qwen/Qwen2.5-7B-Instruct" --qlora

# LoRA로 학습 (24GB+ VRAM)
python scripts/train.py --model "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# 설정 파일 사용 (config에서 model.name 지정)
python scripts/train.py --config configs/training_config.yaml
```

### 3. 평가

```bash
python scripts/evaluate.py --base-model "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter-path outputs/checkpoints/final/adapter
```

### 4. 추론

```bash
# 단일 파일 번역
python scripts/inference.py -m "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter-path outputs/checkpoints/final/adapter \
    --input docs/README.ko.md --output docs/README.en.md

# 대화형 모드
python scripts/inference.py -m "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter-path outputs/checkpoints/final/adapter --interactive

# 디렉토리 일괄 번역
python scripts/inference.py -m "meta-llama/Llama-3.1-8B-Instruct" \
    --adapter-path outputs/checkpoints/final/adapter \
    --input-dir docs/ko --output-dir docs/en
```

### 5. Hugging Face Hub 업로드

학습된 모델을 Hugging Face Hub에 업로드하여 공유할 수 있습니다.

#### 인증 설정

```bash
# 방법 1: CLI 로그인 (권장)
huggingface-cli login

# 방법 2: 환경 변수
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

#### 학습 완료 후 자동 업로드

```bash
# 학습 완료 시 자동으로 Hub에 업로드
python scripts/train.py --model "meta-llama/Llama-3.1-8B-Instruct" --qlora \
    --push-to-hub \
    --hub-repo-id your-username/markdown-translator

# 비공개 리포지토리로 업로드
python scripts/train.py --model "Qwen/Qwen2.5-7B-Instruct" --qlora \
    --push-to-hub \
    --hub-repo-id your-org/internal-model \
    --hub-private
```

#### 수동 업로드

```bash
# 어댑터만 업로드 (용량 작음, 추천)
# base_model은 training_info.json에서 자동으로 읽음
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/markdown-translator

# 병합된 전체 모델 업로드 (단독 사용 가능, --base-model 필수)
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/markdown-translator-merged \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --merge

# 비공개 + 커스텀 설정
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/my-translator \
    --private
```

#### Hub에서 모델 사용하기

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 기본 모델 로드 (학습에 사용한 모델과 동일해야 함)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",  # 학습에 사용한 모델
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "your-username/markdown-translator"
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/markdown-translator"
)
```

## Configuration

### QLoRA 설정 (`configs/qlora_config.yaml`)

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

### 학습 설정 (`configs/training_config.yaml`)

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
| BLEU | N-gram 기반 번역 정확도 |
| chrF | 문자 수준 F-score |
| COMET | 신경망 기반 품질 평가 |
| Preservation Rate | 마크다운 요소 보존율 |

## Hardware Requirements

| Method | VRAM | GPU |
|--------|------|-----|
| QLoRA (4-bit) | 12-16GB | RTX 3090, 4080 |
| LoRA (16-bit) | 20-24GB | RTX 4090, A5000 |
| Full Finetuning | 60GB+ | A100 80GB |

## License

MIT License

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)
