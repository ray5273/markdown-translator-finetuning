#!/usr/bin/env python3
"""모델 평가 스크립트

Usage:
    python scripts/evaluate.py --adapter-path outputs/checkpoints/final/adapter
    python scripts/evaluate.py --config configs/inference_config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from tqdm import tqdm

from src.inference.pipeline import TranslationPipeline
from src.evaluation.metrics import CombinedEvaluator


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Evaluate translation model")

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
        help="LoRA adapter path (overrides config)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Test data file path (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Disable COMET evaluation"
    )

    return parser.parse_args()


def load_test_data(file_path: str, max_samples: int = None):
    """테스트 데이터 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)

                if max_samples and len(data) >= max_samples:
                    break

    return data


def extract_texts(data):
    """데이터에서 텍스트 추출"""
    sources = []
    references = []

    for item in data:
        messages = item['messages']

        # 사용자 메시지에서 한국어 텍스트 추출
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        # 프롬프트 부분 제거하고 실제 한국어 텍스트만 추출
        if '\n\n' in user_msg:
            korean_text = user_msg.split('\n\n', 1)[-1].strip()
        else:
            korean_text = user_msg

        # 어시스턴트 메시지에서 영어 참조 추출
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')

        sources.append(korean_text)
        references.append(assistant_msg)

    return sources, references


def main():
    args = parse_args()

    # 설정 로드
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 경로 설정
    model_config = config.get('model', {})
    base_model = model_config.get('base_model', 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct')
    adapter_path = args.adapter_path or model_config.get('adapter_path')
    test_file = args.test_file or config.get('evaluation', {}).get('test_file', 'data/processed/test.jsonl')

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 테스트 데이터 로드
    print(f"Loading test data from {test_file}")
    test_data = load_test_data(test_file, args.max_samples)
    print(f"Loaded {len(test_data)} test samples")

    sources, references = extract_texts(test_data)

    # 추론 파이프라인 초기화
    print(f"Initializing pipeline...")
    print(f"  Base model: {base_model}")
    print(f"  Adapter: {adapter_path}")

    pipeline = TranslationPipeline(
        base_model_path=base_model,
        adapter_path=adapter_path,
        config_path=args.config
    )

    # 예측 생성
    print("\nGenerating translations...")
    predictions = []

    for source in tqdm(sources, desc="Translating"):
        try:
            translation = pipeline.translate(source)
            predictions.append(translation)
        except Exception as e:
            print(f"Translation failed: {e}")
            predictions.append("")

    # 평가
    print("\nEvaluating...")
    evaluator = CombinedEvaluator(use_comet=not args.no_comet)
    result = evaluator.evaluate(sources, predictions, references)

    # 결과 출력
    evaluator.print_report(result)

    # 결과 저장
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'bleu': result.bleu,
            'chrf': result.chrf,
            'comet': result.comet,
            'preservation_rate': result.preservation_rate,
            'details': result.details,
            'num_samples': len(sources)
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {results_file}")

    # 예측 저장
    predictions_file = output_dir / "predictions.jsonl"
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for src, pred, ref in zip(sources, predictions, references):
            f.write(json.dumps({
                'source': src,
                'prediction': pred,
                'reference': ref
            }, ensure_ascii=False) + '\n')

    print(f"Predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
