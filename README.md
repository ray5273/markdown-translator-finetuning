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

#### 학습 데이터 형식

학습 데이터는 JSONL 형식으로 준비합니다:

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Korean to English translator..."},
    {"role": "user", "content": "Translate the following...\n\n# 프로젝트 소개"},
    {"role": "assistant", "content": "# Project Introduction"}
  ]
}
```

#### 데이터 생성 방법

4가지 방법으로 학습 데이터를 생성할 수 있습니다:

##### 방법 1: 템플릿 기반 생성 (추천 - API 불필요)

```bash
python scripts/generate_synthetic.py --method template --num-samples 1000
```

- **장점**: 무료, 빠름, API 키 불필요
- **출력**: `data/synthetic/template_pairs.jsonl`
- 사전 정의된 마크다운 템플릿으로 다양한 한영 번역 쌍 생성

##### 방법 2: OpenAI API 기반 생성

```bash
export OPENAI_API_KEY="sk-..."
python scripts/generate_synthetic.py --method openai --num-samples 50 --model "gpt-4o"
```

- 고품질 번역 쌍 생성
- 다양한 주제와 스타일 지원

##### 방법 3: Anthropic API 기반 생성

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/generate_synthetic.py --method anthropic --num-samples 50 --model "claude-sonnet-4-20250514"
```

##### 방법 4: 샘플 데이터 생성 (테스트용)

```bash
python scripts/generate_synthetic.py --method sample --num-samples 20
```

- 하드코딩된 샘플 사용 (데모/테스트 목적)

#### 데이터 전처리

생성된 데이터를 학습용으로 전처리하고 train/valid/test로 분할합니다:

```bash
python scripts/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --include-synthetic \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1
```

**출력 파일**:
- `data/processed/train.jsonl` (80%)
- `data/processed/valid.jsonl` (10%)
- `data/processed/test.jsonl` (10%)

#### 전체 파이프라인 예시

```bash
# 1단계: 합성 데이터 생성
python scripts/generate_synthetic.py --method template --num-samples 1000

# 2단계: 전처리 및 분할
python scripts/prepare_data.py \
  --input data/raw \
  --output data/processed \
  --include-synthetic

# 3단계: 학습 시작
python scripts/train.py --config configs/training_config.yaml --qlora
```

#### 데이터 생성 방법 비교

| 방법 | 속도 | 비용 | 품질 | API 필요 |
|------|------|------|------|---------|
| 템플릿 기반 | 매우 빠름 | 무료 | 중간 | ❌ |
| OpenAI | 느림 | 유료 | 높음 | ✅ |
| Anthropic | 느림 | 유료 | 높음 | ✅ |
| 샘플 데이터 | 매우 빠름 | 무료 | 낮음 | ❌ |

#### 마크다운 요소 지원

생성된 데이터는 다양한 마크다운 요소를 포함하여 번역 모델이 마크다운 구조를 잘 보존하도록 학습됩니다.

##### 지원하는 마크다운 요소

| 요소 | 문법 | 설명 |
|------|------|------|
| 헤더 | `# ## ###` | 문서 구조화 |
| 코드 블록 | ` ```language ``` ` | 코드 예제 |
| 인라인 코드 | `` `code` `` | 변수/함수명 |
| 테이블 | `\| col \| col \|` | 데이터 정리 |
| 링크 | `[text](url)` | 외부 참조 |
| 이미지 | `![alt](url)` | 다이어그램 |
| 리스트 | `- item` / `1. item` | 목록 |
| 인용구 | `> quote` | 중요 내용 강조 |
| 굵은 글씨 | `**bold**` | 강조 |
| 기울임 | `*italic*` | 부가 설명 |

##### 스타일별 필수 마크다운 요소

각 문서 스타일마다 필수로 포함해야 하는 마크다운 요소가 정의되어 있습니다:

| 스타일 | 필수 요소 |
|--------|-----------|
| `readme` | 헤더, 코드 블록, 리스트, 링크, 굵은 글씨 |
| `tutorial` | 헤더, 코드 블록, 인라인 코드, 리스트, 링크, 인용구 |
| `api_doc` | 헤더, 코드 블록, 인라인 코드, 테이블, 리스트 |
| `blog` | 헤더, 코드 블록, 링크, 굵은 글씨, 기울임, 인용구 |
| `reference` | 헤더, 테이블, 인라인 코드, 리스트, 링크 |
| `troubleshooting` | 헤더, 코드 블록, 리스트, 인용구, 굵은 글씨 |

##### 생성 데이터 검증

OpenAI/Anthropic API로 생성된 데이터는 자동으로 검증되어 필수 마크다운 요소가 포함되었는지 확인합니다:

```python
# 검증 결과 예시
{
    "is_valid": True,
    "korean_counts": {"headers": 5, "code_blocks": 2, "tables": 1, ...},
    "english_counts": {"headers": 5, "code_blocks": 2, "tables": 1, ...},
    "required_elements": ["headers", "code_blocks", "tables"],
    "missing_korean": [],
    "missing_english": []
}
```

마크다운 보존율 검증 스크립트:

```bash
python scripts/validate_markdown_preservation.py \
    --input data/processed/train.jsonl \
    --min-overall-rate 0.98
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
python scripts/train.py --config configs/training_config.yaml --qlora \
    --push-to-hub \
    --hub-repo-id your-username/exaone-markdown-translator

# 비공개 리포지토리로 업로드
python scripts/train.py --config configs/training_config.yaml --qlora \
    --push-to-hub \
    --hub-repo-id your-org/internal-model \
    --hub-private
```

#### 수동 업로드

```bash
# 어댑터만 업로드 (용량 작음, 추천)
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/exaone-markdown-translator

# 병합된 전체 모델 업로드 (단독 사용 가능)
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/exaone-markdown-translator-merged \
    --merge

# 비공개 + 커스텀 설정
python scripts/upload_to_hub.py \
    --adapter-path outputs/checkpoints/final/adapter \
    --repo-id your-username/my-translator \
    --config configs/hub_config.yaml \
    --private
```

#### Hub에서 모델 사용하기

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(
    base_model,
    "your-username/exaone-markdown-translator"
)
tokenizer = AutoTokenizer.from_pretrained(
    "your-username/exaone-markdown-translator"
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

- [EXAONE 3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [COMET](https://github.com/Unbabel/COMET)
