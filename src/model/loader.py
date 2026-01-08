"""모델 로딩 모듈

EXAONE 3.5-7.8B 모델을 로드하고 LoRA/QLoRA를 적용합니다.
"""

import torch
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

from src.utils.dependencies import require_dependencies

require_dependencies([("yaml", "pyyaml")])

import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)


class ModelLoader:
    """EXAONE 모델 로더

    LoRA/QLoRA 설정을 적용하여 모델을 로드합니다.
    """

    DEFAULT_MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            config_path: YAML 설정 파일 경로
            config: 직접 전달하는 설정 딕셔너리
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            # 기본 설정
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'quantization': {
                'load_in_4bit': True,
                'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_compute_dtype': 'bfloat16',
                'bnb_4bit_use_double_quant': True
            },
            'lora': {
                'r': 64,
                'lora_alpha': 128,
                'lora_dropout': 0.05,
                'bias': 'none',
                'task_type': 'CAUSAL_LM',
                'target_modules': [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'
                ],
                'use_rslora': True
            },
            'model': {
                'use_flash_attention': True,
                'device_map': 'auto'
            }
        }

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 생성"""
        quant_config = self.config.get('quantization', {})

        if not quant_config.get('load_in_4bit', False):
            return None

        compute_dtype = quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
        if compute_dtype == 'bfloat16':
            compute_dtype = torch.bfloat16
        elif compute_dtype == 'float16':
            compute_dtype = torch.float16
        else:
            compute_dtype = torch.float32

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True)
        )

    def _get_lora_config(self) -> LoraConfig:
        """LoRA 설정 생성"""
        lora_config = self.config.get('lora', {})

        return LoraConfig(
            r=lora_config.get('r', 64),
            lora_alpha=lora_config.get('lora_alpha', 128),
            lora_dropout=lora_config.get('lora_dropout', 0.05),
            bias=lora_config.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_config.get('target_modules', [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ]),
            use_rslora=lora_config.get('use_rslora', True)
        )

    def load_tokenizer(
        self,
        model_name: str = None
    ) -> AutoTokenizer:
        """토크나이저 로드

        Args:
            model_name: 모델 이름/경로

        Returns:
            토크나이저
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def load_model(
        self,
        model_name: str = None,
        use_quantization: bool = True
    ) -> AutoModelForCausalLM:
        """모델 로드

        Args:
            model_name: 모델 이름/경로
            use_quantization: 양자화 사용 여부

        Returns:
            모델
        """
        model_name = model_name or self.DEFAULT_MODEL_NAME
        model_config = self.config.get('model', {})

        # 양자화 설정
        bnb_config = self._get_quantization_config() if use_quantization else None

        # Attention 구현 선택
        attn_implementation = None
        if model_config.get('use_flash_attention', True):
            attn_implementation = "flash_attention_2"

        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=model_config.get('device_map', 'auto'),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not use_quantization else None,
            attn_implementation=attn_implementation
        )

        return model

    def prepare_for_training(
        self,
        model: AutoModelForCausalLM,
        use_quantization: bool = True
    ) -> AutoModelForCausalLM:
        """학습을 위한 모델 준비

        양자화된 모델의 경우 kbit training 준비를 수행합니다.

        Args:
            model: 로드된 모델
            use_quantization: 양자화 사용 여부

        Returns:
            준비된 모델
        """
        if use_quantization:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )
        else:
            # Gradient checkpointing 활성화
            model.gradient_checkpointing_enable()

        return model

    def apply_lora(
        self,
        model: AutoModelForCausalLM
    ) -> AutoModelForCausalLM:
        """LoRA 어댑터 적용

        Args:
            model: 모델

        Returns:
            PEFT 모델
        """
        lora_config = self._get_lora_config()

        model = get_peft_model(model, lora_config)

        # 학습 가능한 파라미터 출력
        model.print_trainable_parameters()

        return model

    def load_model_and_tokenizer(
        self,
        model_name: str = None,
        use_quantization: bool = True,
        apply_lora: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """모델과 토크나이저를 함께 로드

        Args:
            model_name: 모델 이름/경로
            use_quantization: 양자화 사용 여부
            apply_lora: LoRA 적용 여부

        Returns:
            (모델, 토크나이저) 튜플
        """
        print(f"Loading tokenizer from {model_name or self.DEFAULT_MODEL_NAME}...")
        tokenizer = self.load_tokenizer(model_name)

        print(f"Loading model...")
        model = self.load_model(model_name, use_quantization)

        print("Preparing model for training...")
        model = self.prepare_for_training(model, use_quantization)

        if apply_lora:
            print("Applying LoRA adapters...")
            model = self.apply_lora(model)

        return model, tokenizer

    @staticmethod
    def load_trained_model(
        base_model_name: str,
        adapter_path: str,
        merge_adapter: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """학습된 모델 로드

        Args:
            base_model_name: 기본 모델 이름
            adapter_path: LoRA 어댑터 경로
            merge_adapter: 어댑터 병합 여부
            device_map: 디바이스 맵
            torch_dtype: 데이터 타입

        Returns:
            (모델, 토크나이저) 튜플
        """
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 기본 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(model, adapter_path)

        # 어댑터 병합 (추론 최적화)
        if merge_adapter:
            model = model.merge_and_unload()

        model.eval()

        return model, tokenizer

    @staticmethod
    def save_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        output_dir: str,
        save_merged: bool = False
    ):
        """모델 저장

        Args:
            model: 저장할 모델
            tokenizer: 토크나이저
            output_dir: 출력 디렉토리
            save_merged: 병합된 모델 저장 여부
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_merged and hasattr(model, 'merge_and_unload'):
            # LoRA를 기본 모델에 병합하여 저장
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_path / "merged")
            tokenizer.save_pretrained(output_path / "merged")
            print(f"Saved merged model to {output_path / 'merged'}")
        else:
            # LoRA 어댑터만 저장
            model.save_pretrained(output_path / "adapter")
            tokenizer.save_pretrained(output_path / "adapter")
            print(f"Saved adapter to {output_path / 'adapter'}")
