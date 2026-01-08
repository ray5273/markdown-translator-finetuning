#!/usr/bin/env python3
"""합성 데이터 생성 스크립트

LLM API 또는 템플릿을 사용하여 학습 데이터를 생성합니다.

Usage:
    # 템플릿 기반 생성 (API 필요 없음)
    python scripts/generate_synthetic.py --method template --num-samples 100

    # OpenAI API 사용
    python scripts/generate_synthetic.py --method openai --num-samples 50

    # Anthropic API 사용
    python scripts/generate_synthetic.py --method anthropic --num-samples 50

    # 샘플 데이터 생성 (데모용)
    python scripts/generate_synthetic.py --method sample --num-samples 20
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 직접 모듈 임포트 (torch 등 무거운 의존성 회피)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "synthetic_generator",
    PROJECT_ROOT / "src" / "data" / "synthetic_generator.py"
)
synthetic_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic_generator)

SyntheticDataGenerator = synthetic_generator.SyntheticDataGenerator
TemplateBasedGenerator = synthetic_generator.TemplateBasedGenerator


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Generate synthetic training data")

    parser.add_argument(
        "--method",
        type=str,
        choices=["template", "openai", "anthropic", "sample"],
        default="template",
        help="Generation method"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/synthetic/{method}_pairs.jsonl)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or use OPENAI_API_KEY/ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (e.g., gpt-4o, claude-sonnet-4-20250514)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def generate_sample_data(num_samples: int, output_path: str):
    """샘플 데이터 생성 (데모/테스트용)

    실제 학습에는 더 다양하고 많은 데이터가 필요합니다.
    """
    samples = [
        {
            "korean": """# 데이터 처리 라이브러리

고성능 데이터 처리를 위한 Python 라이브러리입니다.

## 설치

```bash
pip install dataprocessor
```

## 빠른 시작

```python
from dataprocessor import Processor

proc = Processor()
result = proc.run(data)
print(result)
```

## 주요 기능

- **빠른 처리**: 멀티스레딩 지원
- **메모리 효율**: 스트리밍 처리 가능
- **확장성**: 커스텀 파이프라인 구성

## 문서

자세한 내용은 [공식 문서](https://docs.example.com)를 참조하세요.""",
            "english": """# Data Processing Library

A Python library for high-performance data processing.

## Installation

```bash
pip install dataprocessor
```

## Quick Start

```python
from dataprocessor import Processor

proc = Processor()
result = proc.run(data)
print(result)
```

## Key Features

- **Fast Processing**: Multi-threading support
- **Memory Efficient**: Streaming processing available
- **Extensibility**: Custom pipeline configuration

## Documentation

For more details, see the [official documentation](https://docs.example.com)."""
        },
        {
            "korean": """## API 레퍼런스

### `process_data(input, options=None)`

입력 데이터를 처리하고 결과를 반환합니다.

#### 매개변수

| 이름 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `input` | `str` | 예 | 처리할 입력 데이터 |
| `options` | `dict` | 아니오 | 처리 옵션 |

#### 반환값

- `Result`: 처리 결과 객체

#### 예제

```python
from mylib import process_data

result = process_data("hello world", {"format": "json"})
print(result.output)
```

> **참고**: 이 함수는 스레드 안전합니다.""",
            "english": """## API Reference

### `process_data(input, options=None)`

Processes input data and returns the result.

#### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `input` | `str` | Yes | Input data to process |
| `options` | `dict` | No | Processing options |

#### Returns

- `Result`: Processing result object

#### Example

```python
from mylib import process_data

result = process_data("hello world", {"format": "json"})
print(result.output)
```

> **Note**: This function is thread-safe."""
        },
        {
            "korean": """# 문제 해결 가이드

## 연결 오류

### 증상

`ConnectionError: Unable to connect to server` 오류가 발생합니다.

### 원인

1. 서버가 실행 중이지 않음
2. 방화벽 설정 문제
3. 잘못된 호스트/포트 설정

### 해결 방법

1. 서버 상태 확인:
   ```bash
   systemctl status myserver
   ```

2. 방화벽 규칙 확인:
   ```bash
   sudo ufw status
   ```

3. 설정 파일 확인:
   ```yaml
   server:
     host: localhost
     port: 8080
   ```

문제가 지속되면 [이슈 트래커](https://github.com/example/issues)에 보고해 주세요.""",
            "english": """# Troubleshooting Guide

## Connection Errors

### Symptoms

You receive `ConnectionError: Unable to connect to server` error.

### Causes

1. Server is not running
2. Firewall configuration issues
3. Incorrect host/port settings

### Solutions

1. Check server status:
   ```bash
   systemctl status myserver
   ```

2. Check firewall rules:
   ```bash
   sudo ufw status
   ```

3. Verify configuration file:
   ```yaml
   server:
     host: localhost
     port: 8080
   ```

If the problem persists, please report it on the [issue tracker](https://github.com/example/issues)."""
        },
        {
            "korean": """# 시작하기

## 전제 조건

- Python 3.10 이상
- pip 또는 conda

## 설치

### pip 사용

```bash
pip install mypackage
```

### 소스에서 설치

```bash
git clone https://github.com/example/mypackage.git
cd mypackage
pip install -e .
```

## 설정

환경 변수를 설정하세요:

```bash
export MY_API_KEY="your-api-key"
export MY_DEBUG=true
```

또는 `.env` 파일을 생성하세요:

```
MY_API_KEY=your-api-key
MY_DEBUG=true
```

## 다음 단계

- [튜토리얼](./tutorial.md)을 따라 해보세요
- [API 문서](./api.md)를 확인하세요""",
            "english": """# Getting Started

## Prerequisites

- Python 3.10 or higher
- pip or conda

## Installation

### Using pip

```bash
pip install mypackage
```

### Install from source

```bash
git clone https://github.com/example/mypackage.git
cd mypackage
pip install -e .
```

## Configuration

Set environment variables:

```bash
export MY_API_KEY="your-api-key"
export MY_DEBUG=true
```

Or create a `.env` file:

```
MY_API_KEY=your-api-key
MY_DEBUG=true
```

## Next Steps

- Follow the [tutorial](./tutorial.md)
- Check the [API documentation](./api.md)"""
        },
        {
            "korean": """## 모델 학습

### 데이터 준비

학습 데이터는 다음 형식이어야 합니다:

```json
{
  "text": "입력 텍스트",
  "label": 1
}
```

### 학습 실행

```python
from mymodel import Trainer, Config

config = Config(
    learning_rate=1e-4,
    batch_size=32,
    epochs=10
)

trainer = Trainer(config)
trainer.train(train_data, valid_data)
```

### 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `learning_rate` | 1e-4 | 학습률 |
| `batch_size` | 32 | 배치 크기 |
| `epochs` | 10 | 에폭 수 |

### 평가

```python
metrics = trainer.evaluate(test_data)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```""",
            "english": """## Model Training

### Data Preparation

Training data should be in the following format:

```json
{
  "text": "input text",
  "label": 1
}
```

### Running Training

```python
from mymodel import Trainer, Config

config = Config(
    learning_rate=1e-4,
    batch_size=32,
    epochs=10
)

trainer = Trainer(config)
trainer.train(train_data, valid_data)
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate |
| `batch_size` | 32 | Batch size |
| `epochs` | 10 | Number of epochs |

### Evaluation

```python
metrics = trainer.evaluate(test_data)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```"""
        }
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 샘플 반복하여 num_samples 만큼 생성
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            sample = samples[i % len(samples)].copy()
            sample['metadata'] = {
                'source': 'sample',
                'index': i
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Generated {num_samples} sample pairs, saved to {output_path}")


def main():
    args = parse_args()

    # 출력 경로 설정
    output_path = args.output or f"data/synthetic/{args.method}_pairs.jsonl"

    print(f"\n=== Synthetic Data Generation ===")
    print(f"Method: {args.method}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output: {output_path}")
    print()

    if args.method == "template":
        # 템플릿 기반 생성 (API 불필요)
        generator = TemplateBasedGenerator(seed=args.seed)
        generator.generate_dataset(
            num_samples=args.num_samples,
            output_path=output_path
        )

    elif args.method == "sample":
        # 샘플 데이터 생성
        generate_sample_data(args.num_samples, output_path)

    elif args.method == "openai":
        # OpenAI API 사용
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
            sys.exit(1)

        generator = SyntheticDataGenerator(
            provider="openai",
            api_key=api_key,
            model=args.model
        )
        generator.generate_dataset(
            num_samples=args.num_samples,
            output_path=output_path,
            delay=args.delay
        )

    elif args.method == "anthropic":
        # Anthropic API 사용
        api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: Anthropic API key required. Set ANTHROPIC_API_KEY or use --api-key")
            sys.exit(1)

        generator = SyntheticDataGenerator(
            provider="anthropic",
            api_key=api_key,
            model=args.model
        )
        generator.generate_dataset(
            num_samples=args.num_samples,
            output_path=output_path,
            delay=args.delay
        )

    print("\n=== Generation Complete ===")
    print(f"Output saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review generated data")
    print("  2. Run: python scripts/prepare_data.py --include-synthetic")
    print("  3. Run: python scripts/train.py --qlora")


if __name__ == "__main__":
    main()
