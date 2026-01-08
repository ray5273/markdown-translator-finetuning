"""학습 콜백 모듈

학습 중 사용되는 커스텀 콜백들을 정의합니다.
"""

from pathlib import Path
from typing import Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class SavePeftModelCallback(TrainerCallback):
    """PEFT 모델 저장 콜백

    체크포인트 저장 시 PEFT 어댑터만 저장합니다.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """체크포인트 저장 시 호출"""
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        # PEFT 모델인 경우 어댑터 저장
        model = kwargs.get('model')
        if model is not None and hasattr(model, 'save_pretrained'):
            peft_path = checkpoint_path / "adapter"
            model.save_pretrained(peft_path)
            print(f"Saved PEFT adapter to {peft_path}")

        return control


class LoggingCallback(TrainerCallback):
    """로깅 콜백

    학습 진행 상황을 로깅합니다.
    """

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """로그 기록 시 호출"""
        if state.global_step % self.log_every_n_steps == 0:
            if logs:
                loss = logs.get('loss', 'N/A')
                lr = logs.get('learning_rate', 'N/A')
                print(f"Step {state.global_step}: loss={loss:.4f}, lr={lr:.2e}")

        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """에폭 종료 시 호출"""
        print(f"\n{'='*50}")
        print(f"Epoch {state.epoch:.0f} completed")
        print(f"Total steps: {state.global_step}")
        print(f"{'='*50}\n")

        return control


class EarlyStoppingCallback(TrainerCallback):
    """조기 종료 콜백

    검증 손실이 개선되지 않으면 학습을 조기 종료합니다.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        """
        Args:
            patience: 개선 없이 대기할 에폭 수
            min_delta: 최소 개선 임계값
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs
    ):
        """평가 후 호출"""
        if metrics is None:
            return control

        eval_loss = metrics.get('eval_loss')
        if eval_loss is None:
            return control

        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.counter = 0
            print(f"Validation loss improved to {eval_loss:.4f}")
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} evaluations")

            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} evaluations without improvement")
                control.should_training_stop = True

        return control


class GradientMonitorCallback(TrainerCallback):
    """그래디언트 모니터링 콜백

    그래디언트 노름을 모니터링하여 학습 안정성을 확인합니다.
    """

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """스텝 종료 시 호출"""
        if state.global_step % self.log_every_n_steps == 0:
            model = kwargs.get('model')
            if model is not None:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                print(f"Step {state.global_step}: Gradient norm = {total_norm:.4f}")

        return control
