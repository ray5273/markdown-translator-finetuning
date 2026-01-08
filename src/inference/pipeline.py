"""추론 파이프라인

마크다운 번역을 위한 추론 파이프라인입니다.
"""

import torch
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import yaml
from dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class GenerationConfig:
    """생성 설정"""
    max_new_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    num_beams: int = 1


class TranslationPipeline:
    """마크다운 번역 추론 파이프라인

    학습된 모델을 사용하여 한국어 마크다운을 영어로 번역합니다.
    """

    SYSTEM_PROMPT = """You are an expert Korean to English translator specializing in technical documentation and markdown content.

Your translation guidelines:
1. Produce natural, fluent English while preserving the original meaning
2. Keep ALL markdown formatting exactly as-is:
   - Headers (#, ##, ###)
   - Bold (**text**) and italic (*text*)
   - Code blocks (```language ... ```)
   - Inline code (`code`)
   - Links [text](url) - translate text, keep url unchanged
   - Tables (| ... |)
   - Lists (-, *, 1.)
3. Do NOT translate:
   - Code inside code blocks
   - URLs and file paths
   - Variable names and function names
   - Technical terms that are commonly kept in English
4. Maintain paragraph structure and line breaks"""

    USER_TEMPLATE = """Translate the following Korean markdown content to English. Preserve all markdown formatting exactly.

{korean_text}"""

    def __init__(
        self,
        base_model_path: str = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        adapter_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        merge_adapter: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Args:
            base_model_path: 기본 모델 경로
            adapter_path: LoRA 어댑터 경로 (None이면 기본 모델만 사용)
            config_path: 설정 파일 경로
            device_map: 디바이스 맵
            torch_dtype: 데이터 타입
            merge_adapter: 어댑터 병합 여부 (추론 최적화)
            use_flash_attention: Flash Attention 사용 여부
        """
        self.config = self._load_config(config_path)
        self.gen_config = self._get_generation_config()

        print(f"Loading tokenizer from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model...")
        attn_impl = "flash_attention_2" if use_flash_attention else None

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl
        )

        # LoRA 어댑터 로드
        if adapter_path:
            print(f"Loading adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

            if merge_adapter:
                print("Merging adapter with base model...")
                self.model = self.model.merge_and_unload()

        self.model.eval()
        print("Model loaded successfully!")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """설정 파일 로드"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _get_generation_config(self) -> GenerationConfig:
        """생성 설정 가져오기"""
        gen_config = self.config.get('generation', {})
        return GenerationConfig(
            max_new_tokens=gen_config.get('max_new_tokens', 4096),
            temperature=gen_config.get('temperature', 0.3),
            top_p=gen_config.get('top_p', 0.9),
            top_k=gen_config.get('top_k', 50),
            do_sample=gen_config.get('do_sample', True),
            repetition_penalty=gen_config.get('repetition_penalty', 1.0),
            num_beams=gen_config.get('num_beams', 1)
        )

    def _build_prompt(self, korean_text: str) -> str:
        """프롬프트 구성"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_TEMPLATE.format(korean_text=korean_text)}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    def translate(
        self,
        korean_text: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """한국어 마크다운을 영어로 번역

        Args:
            korean_text: 번역할 한국어 마크다운 텍스트
            generation_config: 생성 설정 (None이면 기본값 사용)

        Returns:
            영어 번역
        """
        config = generation_config or self.gen_config

        # 프롬프트 구성
        prompt = self._build_prompt(korean_text)

        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        input_length = inputs['input_ids'].shape[1]

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                repetition_penalty=config.repetition_penalty,
                num_beams=config.num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # 디코딩 (입력 부분 제외)
        generated_text = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def translate_batch(
        self,
        korean_texts: List[str],
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[str]:
        """배치 번역

        Args:
            korean_texts: 번역할 텍스트 리스트
            batch_size: 배치 크기
            show_progress: 진행률 표시 여부

        Returns:
            번역 리스트
        """
        from tqdm import tqdm

        results = []
        iterator = range(0, len(korean_texts), batch_size)

        if show_progress:
            iterator = tqdm(iterator, desc="Translating")

        for i in iterator:
            batch = korean_texts[i:i + batch_size]
            for text in batch:
                translation = self.translate(text)
                results.append(translation)

        return results

    def translate_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> str:
        """파일 번역

        Args:
            input_path: 입력 파일 경로
            output_path: 출력 파일 경로

        Returns:
            번역된 텍스트
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 파일 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            korean_text = f.read()

        # 번역
        translation = self.translate(korean_text)

        # 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translation)

        print(f"Translation saved to {output_path}")
        return translation


class BatchTranslator:
    """대규모 배치 번역기

    대량의 문서를 효율적으로 번역합니다.
    """

    def __init__(
        self,
        pipeline: TranslationPipeline,
        output_dir: Union[str, Path] = "outputs/translations"
    ):
        """
        Args:
            pipeline: 번역 파이프라인
            output_dir: 출력 디렉토리
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def translate_directory(
        self,
        input_dir: Union[str, Path],
        pattern: str = "*.md",
        recursive: bool = True
    ) -> Dict[str, str]:
        """디렉토리 내 파일들 번역

        Args:
            input_dir: 입력 디렉토리
            pattern: 파일 패턴
            recursive: 재귀 탐색 여부

        Returns:
            {입력 경로: 출력 경로} 매핑
        """
        from tqdm import tqdm

        input_dir = Path(input_dir)

        if recursive:
            files = list(input_dir.rglob(pattern))
        else:
            files = list(input_dir.glob(pattern))

        results = {}

        for file_path in tqdm(files, desc="Translating files"):
            # 상대 경로 유지
            rel_path = file_path.relative_to(input_dir)
            output_path = self.output_dir / rel_path

            try:
                self.pipeline.translate_file(file_path, output_path)
                results[str(file_path)] = str(output_path)
            except Exception as e:
                print(f"Failed to translate {file_path}: {e}")
                results[str(file_path)] = None

        return results
