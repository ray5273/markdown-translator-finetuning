"""커스텀 트레이너 모듈

EXAONE 번역 모델 학습을 위한 커스텀 트레이너입니다.
"""

from typing import Dict, Optional, List, Union
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.trainer_utils import EvalPrediction
import numpy as np


class TranslationTrainer(Trainer):
    """번역 모델 트레이너

    Hugging Face Trainer를 확장하여 번역 태스크에 최적화합니다.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        args: TrainingArguments,
        tokenizer: AutoTokenizer,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        callbacks=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """손실 계산

        Causal LM loss를 계산합니다.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        """체크포인트 저장

        PEFT 모델의 경우 어댑터만 저장합니다.
        """
        # 기본 체크포인트 저장
        super()._save_checkpoint(model, trial, metrics)

        # PEFT 어댑터 별도 저장
        if hasattr(model, 'save_pretrained'):
            checkpoint_folder = f"checkpoint-{self.state.global_step}"
            output_dir = Path(self.args.output_dir) / checkpoint_folder / "adapter"
            model.save_pretrained(output_dir)


def create_training_arguments(
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    **kwargs
) -> TrainingArguments:
    """학습 인자 생성

    Args:
        output_dir: 출력 디렉토리
        num_epochs: 에폭 수
        batch_size: 배치 크기
        gradient_accumulation_steps: 그래디언트 누적 스텝
        learning_rate: 학습률
        warmup_ratio: 워밍업 비율
        logging_steps: 로깅 간격
        save_steps: 저장 간격
        eval_steps: 평가 간격
        bf16: BF16 사용 여부
        gradient_checkpointing: 그래디언트 체크포인팅 사용 여부
        **kwargs: 추가 인자

    Returns:
        TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=kwargs.get('weight_decay', 0.01),
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=kwargs.get('lr_scheduler_type', 'cosine'),
        optim=kwargs.get('optim', 'adamw_8bit'),
        bf16=bf16,
        tf32=kwargs.get('tf32', True),
        logging_dir=kwargs.get('logging_dir', f"{output_dir}/logs"),
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=kwargs.get('save_total_limit', 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=kwargs.get('max_grad_norm', 0.3),
        group_by_length=kwargs.get('group_by_length', True),
        dataloader_num_workers=kwargs.get('dataloader_num_workers', 4),
        dataloader_pin_memory=True,
        report_to=kwargs.get('report_to', 'wandb'),
        run_name=kwargs.get('run_name', 'exaone-markdown-translator'),
        seed=kwargs.get('seed', 42),
    )
