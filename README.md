# EXAONE 3.5 Markdown Translator

EXAONE 3.5-7.8B 모델을 파인튜닝하여 한국어 마크다운 문서를 영어로 번역하는 프로젝트입니다.

## Features

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
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                 # 설정 파일
│   ├── qlora_config.yaml   # QLoRA 설정
│   ├── lora_config.yaml    # LoRA 설정
│   ├── training_config.yaml # 학습 설정
│   └── inference_config.yaml # 추론 설정
├── src/
│   ├── data/               # 데이터 처리
│   ├── model/              # 모델 로딩
│   ├── training/           # 학습 모듈
│   ├── evaluation/         # 평가 메트릭
│   └── inference/          # 추론 파이프라인
├── scripts/
│   ├── train.py            # 학습 스크립트
│   ├── evaluate.py         # 평가 스크립트
│   └── inference.py        # 추론 스크립트
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
# QLoRA로 학습 (12-16GB VRAM)
python scripts/train.py --config configs/training_config.yaml --qlora

# LoRA로 학습 (24GB+ VRAM)
python scripts/train.py --config configs/training_config.yaml
```

### 3. 평가

```bash
python scripts/evaluate.py --adapter-path outputs/checkpoints/final/adapter
```

### 4. 추론

```bash
# 단일 파일 번역
python scripts/inference.py --input docs/README.ko.md --output docs/README.en.md

# 대화형 모드
python scripts/inference.py --interactive

# 디렉토리 일괄 번역
python scripts/inference.py --input-dir docs/ko --output-dir docs/en
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

- [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)
