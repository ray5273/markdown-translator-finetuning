#!/usr/bin/env python3
"""마크다운 보존율 검증 스크립트.

Usage:
    python scripts/validate_markdown_preservation.py \
        --input data/processed/train.jsonl \
        --max-issues 10 \
        --min-overall-rate 0.98 \
        --output outputs/markdown_preservation.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.markdown_parser import MarkdownPreserver
from src.evaluation.metrics import MarkdownPreservationMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Validate markdown preservation in JSONL data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=None,
        help="Fail if number of failed samples is greater than or equal to this value"
    )
    parser.add_argument(
        "--min-overall-rate",
        type=float,
        default=None,
        help="Fail if overall preservation rate is less than or equal to this value"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON output for CI artifacts"
    )
    return parser.parse_args()


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return items


def extract_pair(item):
    messages = item.get("messages", [])
    user_message = next((m.get("content") for m in messages if m.get("role") == "user"), "")
    assistant_message = next(
        (m.get("content") for m in messages if m.get("role") == "assistant"),
        ""
    )
    return user_message, assistant_message


def main():
    args = parse_args()

    preserver = MarkdownPreserver()
    metrics = MarkdownPreservationMetrics()

    data = load_jsonl(args.input)

    results = []
    element_rate_totals = defaultdict(float)
    element_rate_counts = defaultdict(int)
    issue_counter = Counter()

    for index, item in enumerate(data):
        source, translation = extract_pair(item)
        is_valid, details = preserver.validate_preservation(source, translation)
        rates = metrics.compute_preservation_rate(source, translation)

        overall_rate = rates.get("overall_rate", 1.0)
        element_rates = {
            key: value
            for key, value in rates.items()
            if key.endswith("_rate") and key != "overall_rate"
        }

        for key, value in element_rates.items():
            element_rate_totals[key] += value
            element_rate_counts[key] += 1

        for issue in details.get("issues", []):
            issue_counter[issue] += 1

        results.append({
            "index": index,
            "is_valid": is_valid,
            "issues": details.get("issues", []),
            "overall_rate": overall_rate,
            "element_rates": element_rates,
            "source_counts": details.get("source_counts", {}),
            "translated_counts": details.get("translated_counts", {})
        })

    total_samples = len(results)
    failed_samples = sum(1 for result in results if not result["is_valid"])

    if total_samples == 0:
        element_avg_rates = {
            f"{key}_rate": 1.0
            for key in metrics.MARKDOWN_PATTERNS.keys()
        }
        overall_rate_avg = 1.0
    else:
        element_avg_rates = {
            key: (element_rate_totals[key] / element_rate_counts[key])
            if element_rate_counts[key]
            else 1.0
            for key in sorted(element_rate_totals.keys())
        }
        overall_rate_avg = (
            sum(result["overall_rate"] for result in results) / total_samples
        )

    top_issue_examples = [
        {
            "index": result["index"],
            "issues": result["issues"]
        }
        for result in sorted(
            (r for r in results if r["issues"]),
            key=lambda r: len(r["issues"]),
            reverse=True
        )[:5]
    ]

    summary = {
        "total_samples": total_samples,
        "failed_samples": failed_samples,
        "overall_rate_avg": overall_rate_avg,
        "element_avg_rates": element_avg_rates,
        "top_issues": issue_counter.most_common(10),
        "top_issue_examples": top_issue_examples
    }

    print("Markdown Preservation Summary")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Failed samples: {summary['failed_samples']}")
    print(f"Overall average preservation rate: {summary['overall_rate_avg']:.4f}")
    print("Element average preservation rates:")
    for key, value in summary["element_avg_rates"].items():
        print(f"  - {key}: {value:.4f}")
    print("Top issues:")
    for issue, count in summary["top_issues"]:
        print(f"  - {issue} ({count})")
    print("Top issue examples:")
    for example in summary["top_issue_examples"]:
        print(f"  - index {example['index']}: {example['issues']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({
                "summary": summary,
                "results": results
            }, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON report to {output_path}")

    should_fail = False

    if args.max_issues is not None and failed_samples >= args.max_issues:
        print(
            f"\nFailure: failed_samples ({failed_samples}) >= max_issues ({args.max_issues})"
        )
        should_fail = True

    if args.min_overall_rate is not None and overall_rate_avg <= args.min_overall_rate:
        print(
            f"\nFailure: overall_rate_avg ({overall_rate_avg:.4f}) "
            f"<= min_overall_rate ({args.min_overall_rate})"
        )
        should_fail = True

    if should_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
