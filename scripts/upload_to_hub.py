#!/usr/bin/env python3
"""Hugging Face Hub 업로드 스크립트

학습된 모델을 Hugging Face Hub에 업로드합니다.

Usage:
    # 어댑터만 업로드
    python scripts/upload_to_hub.py --adapter-path outputs/checkpoints/final/adapter --repo-id username/model-name

    # 병합된 모델 업로드
    python scripts/upload_to_hub.py --adapter-path outputs/checkpoints/final/adapter --repo-id username/model-name --merge

    # 비공개 리포지토리로 업로드
    python scripts/upload_to_hub.py --adapter-path outputs/checkpoints/final/adapter --repo-id username/model-name --private

    # 설정 파일 사용
    python scripts/upload_to_hub.py --adapter-path outputs/checkpoints/final/adapter --config configs/hub_config.yaml
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dependencies import require_dependencies

require_dependencies([
    ("yaml", "pyyaml"),
    ("huggingface_hub", "huggingface_hub"),
])

import yaml
from src.model.hub_uploader import HubUploader


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="Upload trained model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic upload
    python scripts/upload_to_hub.py \\
        --adapter-path outputs/checkpoints/final/adapter \\
        --repo-id your-username/exaone-markdown-translator

    # Upload merged model
    python scripts/upload_to_hub.py \\
        --adapter-path outputs/checkpoints/final/adapter \\
        --repo-id your-username/exaone-markdown-translator-merged \\
        --merge

    # Private repository with custom config
    python scripts/upload_to_hub.py \\
        --adapter-path outputs/checkpoints/final/adapter \\
        --repo-id your-org/internal-translator \\
        --config configs/hub_config.yaml \\
        --private

Environment Variables:
    HF_TOKEN: Hugging Face API token
    HUGGING_FACE_HUB_TOKEN: Alternative token variable
        """
    )

    # Required arguments
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory"
    )

    # Repository settings
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Hugging Face repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hub_config.yaml",
        help="Hub configuration file path"
    )

    # Upload options
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adapter into base model before upload (creates standalone model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        help="Base model name (required for --merge)"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message"
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Create a pull request instead of direct commit"
    )

    # Training info (optional)
    parser.add_argument(
        "--training-info",
        type=str,
        default=None,
        help="Path to training info JSON file"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to evaluation metrics JSON file"
    )

    # Authentication
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )

    return parser.parse_args()


def load_json_file(path: str) -> dict:
    """JSON 파일 로드"""
    if path and Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def main():
    args = parse_args()

    # 어댑터 경로 확인
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        sys.exit(1)

    # 어댑터 파일 확인
    adapter_files = list(adapter_path.glob("adapter_*"))
    if not adapter_files:
        print(f"Error: No adapter files found in {adapter_path}")
        print("Expected files: adapter_config.json, adapter_model.safetensors or adapter_model.bin")
        sys.exit(1)

    # 설정 로드
    config_path = Path(args.config)
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Using default configuration")
        config = None

    # 토큰 확인
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # 업로더 초기화
    uploader = HubUploader(config=config, token=token)

    # 인증 확인
    print("\nAuthenticating with Hugging Face Hub...")
    if not uploader.authenticate(token):
        print("\nAuthentication failed. Please either:")
        print("  1. Run: huggingface-cli login")
        print("  2. Set HF_TOKEN environment variable")
        print("  3. Use --token argument")
        sys.exit(1)

    # 리포지토리 ID 확인
    repo_id = args.repo_id
    if not repo_id:
        if config and config.get('hub', {}).get('repo_id'):
            repo_id = config['hub']['repo_id']
        else:
            print("\nError: Repository ID is required.")
            print("Specify with --repo-id or in config file.")
            sys.exit(1)

    # 학습 정보 및 메트릭 로드
    training_info = load_json_file(args.training_info)
    metrics = load_json_file(args.metrics)

    # 자동으로 학습 정보 찾기
    if not training_info:
        possible_paths = [
            adapter_path / "training_info.json",
            adapter_path.parent / "training_info.json",
            adapter_path.parent.parent / "training_info.json",
        ]
        for p in possible_paths:
            if p.exists():
                training_info = load_json_file(str(p))
                print(f"Found training info: {p}")
                break

    # 업로드 실행
    print(f"\n{'=' * 60}")
    print(f"Uploading to: {repo_id}")
    print(f"Adapter path: {adapter_path}")
    print(f"Mode: {'Merged model' if args.merge else 'LoRA adapter only'}")
    print(f"Private: {args.private}")
    print(f"{'=' * 60}\n")

    try:
        if args.merge:
            # 병합된 모델 업로드
            repo_url = uploader.upload_merged_model(
                base_model_name=args.base_model,
                adapter_path=str(adapter_path),
                repo_id=repo_id,
                private=args.private,
                commit_message=args.commit_message
            )
        else:
            # 어댑터만 업로드
            repo_url = uploader.upload(
                adapter_path=str(adapter_path),
                repo_id=repo_id,
                private=args.private,
                commit_message=args.commit_message,
                training_info=training_info,
                metrics=metrics,
                create_pr=args.create_pr
            )

        print(f"\n{'=' * 60}")
        print("Upload successful!")
        print(f"{'=' * 60}")
        print(f"\nYour model is available at:")
        print(f"  {repo_url}")
        print(f"\nTo use this model:")
        print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "{args.base_model}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load adapter
model = PeftModel.from_pretrained(model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
""")

    except Exception as e:
        print(f"\nError during upload: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
