#!/usr/bin/env python3
"""합성 데이터 생성 스크립트

LLM API 또는 템플릿을 사용하여 학습 데이터를 생성합니다.

Usage:
    # 템플릿 기반 생성 (API 필요 없음)
    python scripts/generate_synthetic.py --method template --num-samples 100

    # OpenAI API 사용 (순차 처리)
    python scripts/generate_synthetic.py --method openai --num-samples 50

    # OpenAI API 사용 (병렬 처리 - 빠름!)
    python scripts/generate_synthetic.py --method openai --num-samples 1000 --parallel --max-concurrent 50

    # OpenAI Batch API 사용 (50% 비용 절감!)
    python scripts/generate_synthetic.py --method openai --num-samples 1000 --batch --wait

    # Anthropic API 사용
    python scripts/generate_synthetic.py --method anthropic --num-samples 50

    # 샘플 데이터 생성 (데모용)
    python scripts/generate_synthetic.py --method sample --num-samples 20

Performance Comparison (1000 samples):
    - Sequential:  ~7 hours
    - Parallel:    ~15-30 minutes
    - Batch API:   ~1-2 hours (50% cost savings)
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


def load_module(name, path):
    """모듈 동적 로딩"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 기본 생성기 로드
synthetic_generator = load_module(
    "synthetic_generator",
    PROJECT_ROOT / "src" / "data" / "synthetic_generator.py"
)
SyntheticDataGenerator = synthetic_generator.SyntheticDataGenerator
TemplateBasedGenerator = synthetic_generator.TemplateBasedGenerator


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast parallel generation (recommended for speed)
  %(prog)s --method openai --num-samples 1000 --parallel

  # Cost-effective batch generation (50%% cheaper)
  %(prog)s --method openai --num-samples 1000 --batch --wait

  # Template-based (no API needed)
  %(prog)s --method template --num-samples 500
        """
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["template", "openai", "anthropic", "sample"],
        default="template",
        help="Generation method (default: template)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to generate (default: 50)"
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
        help="Delay between API calls in sequential mode (seconds, default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # 병렬 처리 옵션
    parallel_group = parser.add_argument_group('Parallel Processing Options')
    parallel_group.add_argument(
        "--parallel",
        action="store_true",
        help="Enable async parallel processing (faster, same cost)"
    )
    parallel_group.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum concurrent requests for parallel mode (default: 50)"
    )

    # Batch API 옵션
    batch_group = parser.add_argument_group('Batch API Options (OpenAI only)')
    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="Use OpenAI Batch API (50%% cost savings, slower)"
    )
    batch_group.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch completion (default: submit and exit)"
    )
    batch_group.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Status polling interval in seconds for batch mode (default: 30)"
    )
    batch_group.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing batch ID to check status or download results"
    )
    batch_group.add_argument(
        "--batch-action",
        type=str,
        choices=["status", "download", "list", "cancel"],
        default=None,
        help="Batch action: status, download, list, or cancel"
    )

    return parser.parse_args()


def generate_sample_data(num_samples: int, output_path: str):
    """샘플 데이터 생성 (데모/테스트용)"""
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
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            sample = samples[i % len(samples)].copy()
            sample['metadata'] = {
                'source': 'sample',
                'index': i
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"Generated {num_samples} sample pairs, saved to {output_path}")


def run_parallel_generation(args):
    """비동기 병렬 생성 실행"""
    # 비동기 생성기 모듈 로드
    try:
        async_generator = load_module(
            "async_generator",
            PROJECT_ROOT / "src" / "data" / "async_generator.py"
        )
    except Exception as e:
        print(f"Error loading async generator: {e}")
        print("Make sure 'openai' package is installed: pip install openai")
        sys.exit(1)

    output_path = args.output or f"data/synthetic/{args.method}_async_pairs.jsonl"

    api_key = args.api_key or os.getenv(
        "OPENAI_API_KEY" if args.method == "openai" else "ANTHROPIC_API_KEY"
    )
    if not api_key:
        print(f"Error: API key required. Set {'OPENAI_API_KEY' if args.method == 'openai' else 'ANTHROPIC_API_KEY'} or use --api-key")
        sys.exit(1)

    async_generator.run_async_generation(
        provider=args.method,
        num_samples=args.num_samples,
        output_path=output_path,
        api_key=api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )

    return output_path


def run_batch_generation(args):
    """Batch API 생성 실행"""
    if args.method != "openai":
        print("Error: Batch API is only supported for OpenAI")
        sys.exit(1)

    # 배치 생성기 모듈 로드
    try:
        batch_generator = load_module(
            "batch_generator",
            PROJECT_ROOT / "src" / "data" / "batch_generator.py"
        )
    except Exception as e:
        print(f"Error loading batch generator: {e}")
        print("Make sure 'openai' package is installed: pip install openai")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    generator = batch_generator.BatchDataGenerator(
        api_key=api_key,
        model=args.model or "gpt-4o"
    )

    # 배치 액션 처리
    if args.batch_action:
        if args.batch_action == "list":
            generator.list_batches()
            return None

        if not args.batch_id:
            print("Error: --batch-id required for this action")
            sys.exit(1)

        if args.batch_action == "status":
            generator.check_status(args.batch_id)
            return None

        elif args.batch_action == "download":
            output_path = args.output or f"data/synthetic/batch_output.jsonl"
            generator.download_results(args.batch_id, output_path=output_path)
            return output_path

        elif args.batch_action == "cancel":
            generator.cancel_batch(args.batch_id)
            return None

    # 새 배치 작업 제출
    output_path = args.output or f"data/synthetic/batch_output.jsonl"

    batch_id = generator.submit_batch(num_samples=args.num_samples)

    if args.wait:
        # 완료까지 대기
        generator.wait_for_completion(batch_id, poll_interval=args.poll_interval)
        generator.download_results(batch_id, output_path=output_path)
        return output_path
    else:
        print(f"\nBatch job submitted. Use these commands to manage:")
        print(f"  # Check status")
        print(f"  python scripts/generate_synthetic.py --method openai --batch --batch-action status --batch-id {batch_id}")
        print(f"  # Download results (when completed)")
        print(f"  python scripts/generate_synthetic.py --method openai --batch --batch-action download --batch-id {batch_id}")
        return None


def run_sequential_generation(args):
    """순차 생성 실행"""
    output_path = args.output or f"data/synthetic/{args.method}_pairs.jsonl"

    if args.method == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OpenAI API key required. Set OPENAI_API_KEY or use --api-key")
            sys.exit(1)

        generator = SyntheticDataGenerator(
            provider="openai",
            api_key=api_key,
            model=args.model
        )

    elif args.method == "anthropic":
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

    return output_path


def main():
    args = parse_args()

    # 출력 경로 기본값 설정
    if args.output is None:
        if args.parallel:
            args.output = f"data/synthetic/{args.method}_async_pairs.jsonl"
        elif args.batch:
            args.output = f"data/synthetic/batch_output.jsonl"
        else:
            args.output = f"data/synthetic/{args.method}_pairs.jsonl"

    print(f"\n{'='*50}")
    print(f"  Synthetic Data Generation")
    print(f"{'='*50}")
    print(f"  Method: {args.method}")
    print(f"  Samples: {args.num_samples}")
    if args.parallel:
        print(f"  Mode: Parallel (max {args.max_concurrent} concurrent)")
    elif args.batch:
        print(f"  Mode: Batch API (50% cost savings)")
    else:
        print(f"  Mode: Sequential")
    print(f"  Output: {args.output}")
    print(f"{'='*50}\n")

    output_path = None

    if args.method == "template":
        # 템플릿 기반 생성 (API 불필요)
        generator = TemplateBasedGenerator(seed=args.seed)
        generator.generate_dataset(
            num_samples=args.num_samples,
            output_path=args.output
        )
        output_path = args.output

    elif args.method == "sample":
        # 샘플 데이터 생성
        generate_sample_data(args.num_samples, args.output)
        output_path = args.output

    elif args.method in ["openai", "anthropic"]:
        if args.batch:
            # Batch API 사용 (OpenAI only)
            output_path = run_batch_generation(args)
        elif args.parallel:
            # 비동기 병렬 처리
            output_path = run_parallel_generation(args)
        else:
            # 순차 처리 (기존 방식)
            output_path = run_sequential_generation(args)

    if output_path:
        print(f"\n{'='*50}")
        print(f"  Generation Complete!")
        print(f"{'='*50}")
        print(f"  Output: {output_path}")
        print(f"\n  Next steps:")
        print(f"    1. Review generated data")
        print(f"    2. python scripts/prepare_data.py --include-synthetic")
        print(f"    3. python scripts/train.py --qlora")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
