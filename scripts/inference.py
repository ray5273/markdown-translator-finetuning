#!/usr/bin/env python3
"""추론 스크립트

Usage:
    # 단일 파일 번역
    python scripts/inference.py -m "meta-llama/Llama-3.1-8B-Instruct" --input input.md --output output.md

    # 텍스트 직접 입력
    python scripts/inference.py -m "Qwen/Qwen2.5-7B-Instruct" --text "# 안녕하세요"

    # 디렉토리 번역
    python scripts/inference.py --base-model "..." --input-dir docs/ko --output-dir docs/en

    # 대화형 모드
    python scripts/inference.py --base-model "..." --interactive
"""

import os
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dependencies import require_dependencies

require_dependencies([("yaml", "pyyaml")])

import yaml

from src.inference.pipeline import TranslationPipeline, BatchTranslator


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Translate Korean markdown to English")

    parser.add_argument(
        "--base-model", "-m",
        type=str,
        default=None,
        help="Base model HuggingFace ID (overrides config file). "
             "Examples: meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference_config.yaml",
        help="Inference config file path"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter path"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Korean text to translate"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory for batch translation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/translations",
        help="Output directory for batch translation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate"
    )

    return parser.parse_args()


def run_interactive(pipeline):
    """대화형 모드 실행"""
    print("\n" + "=" * 60)
    print("Interactive Translation Mode")
    print("Enter Korean markdown text (type 'quit' to exit)")
    print("Use triple backticks (```) for multi-line input")
    print("=" * 60 + "\n")

    while True:
        try:
            # 입력 받기
            print("Korean> ", end="", flush=True)
            lines = []

            # 첫 줄 읽기
            first_line = input()

            if first_line.lower() == 'quit':
                break

            # 멀티라인 입력 처리
            if first_line.strip() == '```':
                # 멀티라인 모드
                while True:
                    line = input()
                    if line.strip() == '```':
                        break
                    lines.append(line)
                korean_text = '\n'.join(lines)
            else:
                korean_text = first_line

            if not korean_text.strip():
                continue

            # 번역
            print("\nTranslating...\n")
            translation = pipeline.translate(korean_text)

            print("English>")
            print(translation)
            print("\n" + "-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            break


def main():
    args = parse_args()

    # 설정 로드
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # 모델 경로 설정 (CLI 인자 우선)
    model_config = config.get('model', {})
    base_model = args.base_model or model_config.get('base_model')
    adapter_path = args.adapter_path or model_config.get('adapter_path')

    # 모델 미지정시 에러
    if not base_model:
        print("\n" + "=" * 70)
        print("  ERROR: Base model not specified  ".center(70, "!"))
        print("=" * 70)
        print("\nPlease specify a model using one of these methods:")
        print("\n  1. Command line argument:")
        print("     python scripts/inference.py --base-model 'meta-llama/Llama-3.1-8B-Instruct' ...")
        print("\n  2. Config file (configs/inference_config.yaml):")
        print("     model:")
        print("       base_model: 'meta-llama/Llama-3.1-8B-Instruct'")
        print("\nSupported models (examples):")
        print("  - LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
        print("  - meta-llama/Llama-3.1-8B-Instruct")
        print("  - Qwen/Qwen2.5-7B-Instruct")
        print("=" * 70 + "\n")
        sys.exit(1)

    # 파이프라인 초기화
    print(f"Loading model: {base_model}")
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")

    pipeline = TranslationPipeline(
        base_model_path=base_model,
        adapter_path=adapter_path,
        config_path=args.config
    )

    # 생성 설정 오버라이드
    if args.temperature is not None:
        pipeline.gen_config.temperature = args.temperature
    if args.max_tokens is not None:
        pipeline.gen_config.max_new_tokens = args.max_tokens

    # 실행 모드 결정
    if args.interactive:
        run_interactive(pipeline)

    elif args.text:
        # 직접 텍스트 번역
        print("\nTranslating...")
        translation = pipeline.translate(args.text)
        print("\n" + "=" * 40)
        print("Translation:")
        print("=" * 40)
        print(translation)

    elif args.input:
        # 파일 번역
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        output_path = args.output or input_path.with_suffix('.en.md')
        pipeline.translate_file(input_path, output_path)

    elif args.input_dir:
        # 디렉토리 번역
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        batch_translator = BatchTranslator(pipeline, output_dir=args.output_dir)
        results = batch_translator.translate_directory(input_dir)

        print(f"\nTranslated {len([v for v in results.values() if v])} files")

    else:
        print("No input specified. Use --text, --input, --input-dir, or --interactive")
        print("Run with --help for more information")


if __name__ == "__main__":
    main()
