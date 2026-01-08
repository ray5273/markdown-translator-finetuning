#!/usr/bin/env python3
"""EXAONE 3.5-7.8B 마크다운 번역 파인튜닝 학습 스크립트

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --qlora
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
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dotenv import load_dotenv

# 프로젝트 모듈
from src.model.loader import ModelLoader
from src.data.dataset import TranslationDataset, TranslationDatasetWithMasking
from src.training.callbacks import SavePeftModelCallback, LoggingCallback, EarlyStoppingCallback
from src.training.collator import DataCollatorForCausalLM

# 환경 변수 로드
load_dotenv()


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Train EXAONE markdown translator")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Training config file path"
    )
    parser.add_argument(
        "--qlora-config",
        type=str,
        default="configs/qlora_config.yaml",
        help="QLoRA config file path"
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default="configs/lora_config.yaml",
        help="LoRA config file path"
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization)"
    )
    parser.add_argument(
        "--use-masking",
        action="store_true",
        default=True,
        help="Use prompt masking for training"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )

    # Hub upload options
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained model to Hugging Face Hub after training"
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Make the Hub repository private"
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_wandb(config: dict, args):
    """WandB 설정"""
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        return

    try:
        import wandb

        wandb_config = config.get('wandb', {})
        project = args.wandb_project or wandb_config.get('project', 'exaone-markdown-translator')

        wandb.init(
            project=project,
            entity=wandb_config.get('entity'),
            tags=wandb_config.get('tags', []),
            config=config
        )
        print(f"WandB initialized: project={project}")

    except ImportError:
        print("WandB not installed. Disabling logging.")
        os.environ["WANDB_DISABLED"] = "true"


def main():
    args = parse_args()

    # 설정 로드
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # LoRA/QLoRA 설정 선택
    if args.qlora:
        print("Using QLoRA (4-bit quantization)")
        lora_config_path = args.qlora_config
    else:
        print("Using LoRA (16-bit)")
        lora_config_path = args.lora_config

    # WandB 설정
    setup_wandb(config, args)

    # 모델 로더 초기화
    print(f"Loading model configuration from {lora_config_path}")
    loader = ModelLoader(config_path=lora_config_path)

    # 모델 및 토크나이저 로드
    model_name = config['model']['name']
    print(f"Loading model: {model_name}")

    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=model_name,
        use_quantization=args.qlora,
        apply_lora=True
    )

    print(f"Model loaded. Device: {next(model.parameters()).device}")

    # 데이터셋 로드
    print("Loading datasets...")
    data_config = config['data']

    DatasetClass = TranslationDatasetWithMasking if args.use_masking else TranslationDataset

    train_dataset = DatasetClass(
        data_path=data_config['train_file'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    print(f"Train dataset: {len(train_dataset)} examples")

    eval_dataset = DatasetClass(
        data_path=data_config['valid_file'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    print(f"Eval dataset: {len(eval_dataset)} examples")

    # 데이터 콜레이터
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # 학습 인자
    train_config = config['training']
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        lr_scheduler_type=train_config['lr_scheduler_type'],
        optim=train_config['optim'],
        bf16=train_config['bf16'],
        tf32=train_config.get('tf32', True),
        logging_dir=train_config.get('logging_dir', f"{train_config['output_dir']}/logs"),
        logging_steps=train_config['logging_steps'],
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        eval_strategy=train_config['eval_strategy'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=train_config['save_total_limit'],
        load_best_model_at_end=train_config['load_best_model_at_end'],
        metric_for_best_model=train_config['metric_for_best_model'],
        greater_is_better=train_config.get('greater_is_better', False),
        gradient_checkpointing=train_config['gradient_checkpointing'],
        max_grad_norm=train_config['max_grad_norm'],
        group_by_length=train_config.get('group_by_length', True),
        dataloader_num_workers=train_config.get('dataloader_num_workers', 4),
        dataloader_pin_memory=True,
        report_to="none" if args.no_wandb else train_config.get('report_to', 'wandb'),
        run_name=train_config.get('run_name', 'exaone-markdown-translator'),
    )

    # 콜백 설정
    callbacks = [
        SavePeftModelCallback(),
        LoggingCallback(log_every_n_steps=train_config['logging_steps'] * 10),
    ]

    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    # 학습 시작
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # 최종 모델 저장
    print("\nSaving final model...")
    final_output_dir = Path(train_config['output_dir']) / "final"
    final_adapter_path = final_output_dir / "adapter"
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"Model saved to {final_adapter_path}")

    # 학습 정보 저장 (Hub 업로드용)
    training_info = {
        "base_model": model_name,
        "num_train_epochs": train_config['num_train_epochs'],
        "per_device_train_batch_size": train_config['per_device_train_batch_size'],
        "gradient_accumulation_steps": train_config['gradient_accumulation_steps'],
        "learning_rate": train_config['learning_rate'],
        "lr_scheduler_type": train_config['lr_scheduler_type'],
        "warmup_ratio": train_config['warmup_ratio'],
        "use_qlora": args.qlora,
        "use_masking": args.use_masking,
    }
    import json
    with open(final_output_dir / "training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    # Hugging Face Hub 업로드
    if args.push_to_hub:
        print("\n" + "=" * 60)
        print("Uploading to Hugging Face Hub...")
        print("=" * 60)

        try:
            from src.model.hub_uploader import HubUploader

            if not args.hub_repo_id:
                print("Error: --hub-repo-id is required when using --push-to-hub")
            else:
                uploader = HubUploader(token=args.hub_token)

                if uploader.authenticate(args.hub_token):
                    repo_url = uploader.upload(
                        adapter_path=str(final_adapter_path),
                        repo_id=args.hub_repo_id,
                        private=args.hub_private,
                        training_info=training_info
                    )
                    print(f"\nModel uploaded to: {repo_url}")
                else:
                    print("Hub authentication failed. Model saved locally only.")

        except Exception as e:
            print(f"Hub upload failed: {e}")
            print("Model saved locally. You can upload manually using:")
            print(f"  python scripts/upload_to_hub.py --adapter-path {final_adapter_path} --repo-id YOUR_REPO_ID")

    # WandB 종료
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
