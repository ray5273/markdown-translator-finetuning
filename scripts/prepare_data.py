#!/usr/bin/env python3
"""데이터 준비 스크립트

수집된 데이터를 학습용 형식으로 변환합니다.

Usage:
    # 전체 파이프라인
    python scripts/prepare_data.py --input data/raw --output data/processed

    # 특정 파일만 처리
    python scripts/prepare_data.py --input data/raw/collected_data.jsonl --output data/processed

    # 합성 데이터 포함
    python scripts/prepare_data.py --input data/raw --output data/processed --include-synthetic
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.data.preprocessor import DataPreprocessor, TranslationExample
from src.data.markdown_parser import MarkdownPreserver


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Prepare training data")

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Input directory or file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output directory"
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic data from data/synthetic"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation data ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test data ratio"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum text length (characters)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum text length (characters)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def load_jsonl(file_path: Path) -> list:
    """JSONL 파일 로드"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line in {file_path}: {e}")
    return data


def load_all_data(input_path: Path, include_synthetic: bool = False) -> list:
    """모든 데이터 로드"""
    all_data = []

    if input_path.is_file():
        # 단일 파일
        print(f"Loading from {input_path}")
        all_data.extend(load_jsonl(input_path))

    elif input_path.is_dir():
        # 디렉토리의 모든 JSONL 파일
        for jsonl_file in input_path.glob("**/*.jsonl"):
            print(f"Loading from {jsonl_file}")
            all_data.extend(load_jsonl(jsonl_file))

        # JSON 파일도 로드
        for json_file in input_path.glob("**/*.json"):
            print(f"Loading from {json_file}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    elif isinstance(data, dict) and 'data' in data:
                        all_data.extend(data['data'])
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")

    # 합성 데이터 포함
    if include_synthetic:
        synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
        if synthetic_dir.exists():
            for jsonl_file in synthetic_dir.glob("*.jsonl"):
                print(f"Loading synthetic data from {jsonl_file}")
                all_data.extend(load_jsonl(jsonl_file))

    print(f"Total raw data loaded: {len(all_data)}")
    return all_data


def convert_to_examples(raw_data: list) -> list:
    """원시 데이터를 TranslationExample로 변환"""
    examples = []

    for item in raw_data:
        korean = None
        english = None
        metadata = {}

        # 다양한 형식 지원
        if 'korean' in item and 'english' in item:
            korean = item['korean']
            english = item['english']
            metadata = item.get('metadata', {})

        elif 'ko' in item and 'en' in item:
            korean = item['ko']
            english = item['en']

        elif 'src' in item and 'tgt' in item:
            korean = item['src']
            english = item['tgt']

        elif 'source' in item and 'target' in item:
            korean = item['source']
            english = item['target']

        if korean and english:
            examples.append(TranslationExample(
                korean=korean,
                english=english,
                metadata=metadata
            ))

    print(f"Converted to {len(examples)} examples")
    return examples


def analyze_data(examples: list):
    """데이터 분석 및 통계 출력"""
    print("\n=== Data Analysis ===")

    # 길이 통계
    ko_lengths = [len(e.korean) for e in examples]
    en_lengths = [len(e.english) for e in examples]

    print(f"Korean text length - Min: {min(ko_lengths)}, Max: {max(ko_lengths)}, "
          f"Avg: {sum(ko_lengths)/len(ko_lengths):.1f}")
    print(f"English text length - Min: {min(en_lengths)}, Max: {max(en_lengths)}, "
          f"Avg: {sum(en_lengths)/len(en_lengths):.1f}")

    # 마크다운 요소 분석
    md_parser = MarkdownPreserver()
    total_counts = {}

    for example in examples[:100]:  # 샘플만 분석
        counts = md_parser.analyze_markdown(example.korean)
        for key, value in counts.items():
            total_counts[key] = total_counts.get(key, 0) + value

    print("\nMarkdown elements (sample of 100):")
    for key, value in sorted(total_counts.items(), key=lambda x: -x[1]):
        if value > 0:
            print(f"  {key}: {value}")

    print("=" * 30 + "\n")


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드
    print("\n=== Loading Data ===")
    raw_data = load_all_data(input_path, args.include_synthetic)

    if not raw_data:
        print("No data found. Please collect data first using scripts/generate_synthetic.py "
              "or place data in data/raw/")
        return

    # TranslationExample로 변환
    examples = convert_to_examples(raw_data)

    if not examples:
        print("No valid examples found.")
        return

    # 데이터 분석
    analyze_data(examples)

    # 전처리기 초기화
    preprocessor = DataPreprocessor(
        max_length=args.max_length,
        min_length=args.min_length,
        use_template_variety=True,
        seed=args.seed
    )

    # 전처리 및 분할, 저장
    print("\n=== Processing and Saving ===")
    stats = preprocessor.process_and_save(
        data=examples,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    # 결과 출력
    print("\n=== Final Statistics ===")
    print(f"Total processed: {stats['total']}")
    print(f"Train set: {stats['train']}")
    print(f"Valid set: {stats['valid']}")
    print(f"Test set: {stats['test']}")
    print(f"\nOutput directory: {output_dir}")

    # 샘플 출력
    print("\n=== Sample Data ===")
    train_file = output_dir / "train.jsonl"
    with open(train_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())

    print("System prompt (first 200 chars):")
    print(sample['messages'][0]['content'][:200] + "...")
    print("\nUser message (first 300 chars):")
    print(sample['messages'][1]['content'][:300] + "...")
    print("\nAssistant response (first 300 chars):")
    print(sample['messages'][2]['content'][:300] + "...")


if __name__ == "__main__":
    main()
