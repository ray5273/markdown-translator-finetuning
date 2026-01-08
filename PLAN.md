# EXAONE 3.5-7.8B 마크다운 한영 번역 파인튜닝 프로젝트 계획

## 1. 프로젝트 개요

**목표**: EXAONE 3.5-7.8B-Instruct 모델을 파인튜닝하여 마크다운 형식을 보존하면서 한국어 → 영어 번역을 수행하는 모델 개발

> **참고**: 사용자가 언급한 "8B"는 실제로 **LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct** 모델입니다 (7.8B 파라미터).

---

## 2. 프로젝트 구조

```
markdown-translator-finetuning/
├── PLAN.md                          # 프로젝트 계획 문서
├── README.md                        # 프로젝트 설명
├── requirements.txt                 # Python 의존성
├── pyproject.toml                   # 프로젝트 설정
│
├── configs/                         # 설정 파일들
│   ├── lora_config.yaml            # LoRA 하이퍼파라미터
│   ├── qlora_config.yaml           # QLoRA 하이퍼파라미터
│   ├── training_config.yaml        # 학습 설정
│   └── inference_config.yaml       # 추론 설정
│
├── data/                            # 데이터 디렉토리
│   ├── raw/                         # 원본 데이터
│   │   ├── parallel_corpus/        # 한영 병렬 코퍼스
│   │   └── markdown_samples/       # 마크다운 샘플
│   ├── processed/                   # 전처리된 데이터
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── test.jsonl
│   └── synthetic/                   # 합성 데이터
│       └── generated_pairs.jsonl
│
├── src/                             # 소스 코드
│   ├── __init__.py
│   ├── data/                        # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── collector.py            # 데이터 수집
│   │   ├── preprocessor.py         # 전처리 파이프라인
│   │   ├── markdown_parser.py      # 마크다운 파싱/보존
│   │   ├── dataset.py              # PyTorch Dataset 클래스
│   │   └── synthetic_generator.py  # 합성 데이터 생성
│   │
│   ├── model/                       # 모델 관련 모듈
│   │   ├── __init__.py
│   │   ├── loader.py               # 모델 로딩 (LoRA/QLoRA)
│   │   ├── lora_config.py          # LoRA 설정
│   │   └── quantization.py         # 양자화 설정
│   │
│   ├── training/                    # 학습 모듈
│   │   ├── __init__.py
│   │   ├── trainer.py              # 커스텀 트레이너
│   │   ├── callbacks.py            # 학습 콜백
│   │   └── collator.py             # 데이터 콜레이터
│   │
│   ├── evaluation/                  # 평가 모듈
│   │   ├── __init__.py
│   │   ├── metrics.py              # BLEU, COMET, chrF
│   │   ├── markdown_eval.py        # 마크다운 보존율 평가
│   │   └── evaluator.py            # 통합 평가기
│   │
│   └── inference/                   # 추론 모듈
│       ├── __init__.py
│       ├── pipeline.py             # 추론 파이프라인
│       └── batch_translator.py     # 배치 번역기
│
├── scripts/                         # 실행 스크립트
│   ├── prepare_data.py             # 데이터 준비
│   ├── generate_synthetic.py       # 합성 데이터 생성
│   ├── train.py                    # 학습 실행
│   ├── evaluate.py                 # 평가 실행
│   └── inference.py                # 추론 실행
│
├── notebooks/                       # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_evaluation_results.ipynb
│
├── tests/                           # 테스트
│   ├── test_data_processing.py
│   ├── test_markdown_parser.py
│   └── test_evaluation.py
│
└── outputs/                         # 출력 디렉토리
    ├── checkpoints/                # 모델 체크포인트
    ├── logs/                       # 학습 로그
    └── results/                    # 평가 결과
```

---

## 3. 데이터셋 준비

### 3.1 데이터 소스

#### 공개 한영 병렬 코퍼스
1. **AI Hub 한영 번역 코퍼스**: 뉴스, 문화 데이터 포함 (수동 다운로드 필요)
2. **UKren Corpus**: 125만 문장 쌍 (University of Ulsan)
3. **OPUS (opus.nlpl.eu)**: 다양한 도메인의 병렬 데이터

#### 마크다운 특화 데이터 생성
1. **GitHub 문서**: 한국어 README, 문서 파일 수집
2. **기술 블로그**: 개발 관련 한국어 포스트
3. **합성 데이터**: GPT-4/Claude를 활용한 마크다운 번역 쌍 생성

### 3.2 마크다운 구조 보존 전략

번역 시 보존해야 할 마크다운 요소:
- 코드 블록 (```language ... ```) → 플레이스홀더 `{{CODE_BLOCK_N}}`로 치환
- 인라인 코드 (`code`) → `{{INLINE_CODE_N}}`
- URL → `{{URL_N}}` (링크 텍스트는 번역)
- 이미지 → `{{IMAGE_N}}`
- 테이블 구조, 헤더, 리스트 형식 유지

### 3.3 데이터 형식 (JSONL)

```json
{
  "messages": [
    {"role": "system", "content": "You are an expert Korean to English translator..."},
    {"role": "user", "content": "Translate the following Korean markdown...\n\n# 프로젝트 소개\n\n이 프로젝트는 **머신러닝** 기반입니다.\n\n```python\nprint('Hello')\n```"},
    {"role": "assistant", "content": "# Project Introduction\n\nThis project is **machine learning**-based.\n\n```python\nprint('Hello')\n```"}
  ],
  "metadata": {"has_code_block": true, "has_table": false}
}
```

---

## 4. 파인튜닝 방법론

### 4.1 방법 비교

| 방법 | VRAM 요구량 | 학습 속도 | 품질 | 권장 상황 |
|------|------------|----------|------|----------|
| **Full Finetuning** | ~60GB+ | 느림 | 최상 | A100 80GB 다수 보유 시 |
| **LoRA (16-bit)** | ~20-24GB | 빠름 | 우수 | RTX 4090, A5000 |
| **QLoRA (4-bit)** | ~12-16GB | 중간 | 양호 | RTX 3090, 4080 |

### 4.2 권장: QLoRA

EXAONE 3.5-7.8B 모델의 경우 **QLoRA**를 권장:

- 7.8B 모델 16-bit = ~15.6GB → QLoRA 4-bit = ~4GB로 압축
- 단일 RTX 4090 (24GB) 또는 A5000에서 학습 가능
- 번역 태스크에서 Full Finetuning 대비 90%+ 성능 달성 가능

### 4.3 하드웨어 요구사항

| 설정 | 최소 VRAM | 권장 GPU | Batch Size |
|------|----------|---------|------------|
| QLoRA 4-bit | 12GB | RTX 3090/4080 | 1-2 |
| QLoRA 4-bit (최적) | 24GB | RTX 4090/A5000 | 4-8 |
| LoRA 16-bit | 24GB | A5000/A6000 | 2-4 |
| Full Finetuning | 80GB+ | A100 80GB x2+ | 8+ |

---

## 5. 학습 설정

### 5.1 LoRA 하이퍼파라미터

```yaml
lora:
  r: 64                    # Rank (64-128 권장 for translation)
  lora_alpha: 128          # Alpha = 2 * r
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

  # EXAONE 모델의 타겟 모듈
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

  use_rslora: true         # RSLoRA 사용 (안정적인 학습)
```

### 5.2 QLoRA 양자화 설정

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"          # Normal Float 4-bit
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true     # 중첩 양자화 (메모리 절약)
```

### 5.3 학습 하이퍼파라미터

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8      # effective batch size = 16

  learning_rate: 2.0e-4               # LoRA 권장 학습률
  weight_decay: 0.01
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  optim: "adamw_8bit"                 # 8-bit Adam (메모리 절약)

  bf16: true
  tf32: true
  gradient_checkpointing: true        # 메모리 절약
  max_grad_norm: 0.3
```

---

## 6. 프롬프트 템플릿

### 6.1 시스템 프롬프트

```
You are an expert Korean to English translator specializing in technical documentation and markdown content.

Your translation guidelines:
1. Produce natural, fluent English while preserving the original meaning
2. Keep ALL markdown formatting exactly as-is:
   - Headers (#, ##, ###)
   - Bold (**text**) and italic (*text*)
   - Code blocks (```language ... ```)
   - Inline code (`code`)
   - Links [text](url) - translate text, keep url
   - Tables (| ... |)
   - Lists (-, *, 1.)
3. Do NOT translate:
   - Code inside code blocks
   - URLs and file paths
   - Variable names and function names
   - Technical terms commonly kept in English
4. Maintain paragraph structure and line breaks
```

### 6.2 사용자 프롬프트 템플릿

```
Translate the following Korean markdown document to English:

---
{korean_text}
---

Provide only the English translation, preserving all markdown formatting:
```

---

## 7. 평가 메트릭

### 7.1 번역 품질 평가

| 메트릭 | 설명 | 용도 |
|--------|------|------|
| **BLEU** | N-gram 기반 정밀도 | 기본 번역 품질 |
| **chrF** | 문자 수준 F-score | 한국어처럼 형태소가 복잡한 언어에 효과적 |
| **COMET** | 신경망 기반 품질 평가 | 인간 판단과 높은 상관관계 |

### 7.2 마크다운 보존 평가

- 헤더 보존율
- 코드 블록 보존율
- 링크/이미지 보존율
- 테이블 구조 보존율
- 리스트 형식 보존율
- **전체 마크다운 보존율**

---

## 8. 의존성

```txt
# Core
torch>=2.1.0
transformers>=4.43.0
datasets>=2.14.0
accelerate>=0.24.0

# PEFT & Quantization
peft>=0.7.0
bitsandbytes>=0.41.0

# Evaluation
sacrebleu>=2.3.0
unbabel-comet>=2.2.0
evaluate>=0.4.0

# Data Processing
pandas>=2.0.0
tiktoken>=0.5.0
mistune>=3.0.0  # Markdown parsing

# Experiment Tracking
wandb>=0.15.0

# Utils
pyyaml>=6.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

---

## 9. 구현 로드맵

### Phase 1: 환경 설정
- [ ] 프로젝트 구조 생성
- [ ] 의존성 설치 및 환경 구성
- [ ] EXAONE 3.5-7.8B 모델 다운로드 및 테스트

### Phase 2: 데이터 준비
- [ ] 공개 한영 병렬 코퍼스 수집 (AI Hub, UKren)
- [ ] 마크다운 특화 데이터 생성/수집
- [ ] 합성 데이터 생성 파이프라인 구축
- [ ] 전처리 파이프라인 구현
- [ ] 데이터셋 분할 (train/valid/test)

### Phase 3: 모델 학습
- [ ] QLoRA 설정 최적화
- [ ] 학습 스크립트 실행
- [ ] 하이퍼파라미터 튜닝
- [ ] 체크포인트 관리

### Phase 4: 평가 및 최적화
- [ ] 번역 품질 평가 (BLEU, COMET, chrF)
- [ ] 마크다운 보존율 평가
- [ ] 오류 분석 및 데이터 보강
- [ ] 추가 학습 (필요시)

### Phase 5: 배포 준비
- [ ] 추론 파이프라인 최적화
- [ ] 모델 병합 (merge_and_unload)
- [ ] 문서화 및 사용 가이드 작성

---

## 10. 핵심 구현 파일

| 파일 | 역할 |
|------|------|
| `src/data/markdown_parser.py` | 마크다운 구조 보존 처리의 핵심 |
| `src/data/preprocessor.py` | 데이터 전처리 및 학습 형식 변환 |
| `src/model/loader.py` | EXAONE 모델 로딩 및 LoRA/QLoRA 적용 |
| `src/data/dataset.py` | PyTorch Dataset 및 프롬프트 템플릿 |
| `scripts/train.py` | 학습 실행 스크립트 |
| `src/evaluation/metrics.py` | BLEU, COMET, chrF 평가 |
| `src/inference/pipeline.py` | 추론 파이프라인 |

---

## 11. 참고 자료

- [EXAONE 3.5 Official Repository](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [Unsloth LoRA Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)
- [AI Hub Ko-En Parallel Corpus](https://ko-nlp.github.io/Korpora/en-docs/corpuslist/aihub_translation.html)
- [COMET MT Evaluation](https://github.com/Unbabel/COMET)
