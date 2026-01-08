"""Hugging Face Hub 업로드 모듈

학습된 모델을 Hugging Face Hub에 업로드합니다.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

from src.utils.dependencies import require_dependencies

require_dependencies([
    ("yaml", "pyyaml"),
    ("huggingface_hub", "huggingface_hub"),
])

import yaml
from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    login,
    whoami,
    ModelCard,
    ModelCardData,
)
from transformers import AutoTokenizer


class HubUploader:
    """Hugging Face Hub 업로더

    학습된 LoRA 어댑터 또는 병합된 모델을 Hub에 업로드합니다.
    """

    DEFAULT_CONFIG_PATH = "configs/hub_config.yaml"

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None
    ):
        """
        Args:
            config_path: Hub 설정 파일 경로
            config: 직접 전달하는 설정 딕셔너리
            token: Hugging Face API 토큰 (없으면 환경변수/캐시 사용)
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()

        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        self.api = HfApi(token=self.token)
        self._authenticated = False

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 설정 반환"""
        return {
            'hub': {
                'repo_id': None,
                'private': False,
                'commit_message': "Upload fine-tuned model",
                'branch': "main",
                'create_repo': True
            },
            'model_card': {
                'language': ['ko', 'en'],
                'license': 'apache-2.0',
                'tags': ['translation', 'korean', 'english', 'lora', 'peft'],
                'pipeline_tag': 'text-generation',
                'base_model': None  # Will be set from training info or CLI
            },
            'upload': {
                'mode': 'adapter',
                'include_tokenizer': True,
                'include_config': True,
                'include_inference_example': True
            }
        }

    def authenticate(self, token: Optional[str] = None) -> bool:
        """Hugging Face Hub 인증

        Args:
            token: API 토큰 (없으면 대화형 로그인)

        Returns:
            인증 성공 여부
        """
        token = token or self.token

        try:
            if token:
                login(token=token)
            else:
                # 대화형 로그인 또는 캐시된 토큰 사용
                login()

            # 인증 확인
            user_info = whoami()
            print(f"Authenticated as: {user_info['name']}")
            self._authenticated = True
            return True

        except Exception as e:
            print(f"Authentication failed: {e}")
            print("Please run `huggingface-cli login` or set HF_TOKEN environment variable.")
            return False

    def _ensure_authenticated(self):
        """인증 상태 확인"""
        if not self._authenticated:
            try:
                whoami(token=self.token)
                self._authenticated = True
            except Exception:
                if not self.authenticate():
                    raise RuntimeError("Hugging Face Hub authentication required.")

    def create_model_card(
        self,
        repo_id: str,
        output_dir: Path,
        training_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """모델 카드 생성

        Args:
            repo_id: 리포지토리 ID
            output_dir: 출력 디렉토리
            training_info: 학습 정보 (에폭, 배치 크기 등)
            metrics: 평가 메트릭 (BLEU, chrF 등)

        Returns:
            모델 카드 내용
        """
        card_config = self.config.get('model_card', {})

        # 모델 카드 데이터
        card_data = ModelCardData(
            language=card_config.get('language', ['ko', 'en']),
            license=card_config.get('license', 'apache-2.0'),
            tags=card_config.get('tags', []),
            pipeline_tag=card_config.get('pipeline_tag', 'text-generation'),
            base_model=card_config.get('base_model'),
            datasets=card_config.get('datasets', [])
        )

        # 기본 설명
        description = card_config.get('description', '')

        # 학습 정보 추가
        if training_info:
            description += "\n\n## Training Hyperparameters\n\n"
            description += "| Parameter | Value |\n|-----------|-------|\n"
            for key, value in training_info.items():
                description += f"| {key} | {value} |\n"

        # 메트릭 추가
        if metrics:
            description += "\n\n## Evaluation Results\n\n"
            description += "| Metric | Score |\n|--------|-------|\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    description += f"| {key} | {value:.4f} |\n"
                else:
                    description += f"| {key} | {value} |\n"

        # repo_id 치환
        description = description.replace("YOUR_USERNAME/YOUR_MODEL_NAME", repo_id)

        # 모델 카드 생성
        card = ModelCard.from_template(
            card_data,
            template_str=f"---\n{{{{ card_data }}}}\n---\n\n{description}"
        )

        # 파일로 저장
        card_path = output_dir / "README.md"
        card.save(str(card_path))

        return str(card)

    def _create_inference_example(self, repo_id: str, output_dir: Path, base_model: Optional[str] = None):
        """추론 예제 코드 생성"""
        base_model_str = base_model or "YOUR_BASE_MODEL"
        example_code = f'''"""Markdown Translator - Inference Example

이 파일은 {repo_id} 모델을 사용하여 추론하는 예제입니다.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_name: str = "{base_model_str}", merge_adapter: bool = False):
    """모델 로드

    Args:
        base_model_name: 기본 모델 이름
        merge_adapter: 어댑터를 기본 모델에 병합할지 여부
                       True면 추론 속도가 빨라지지만 메모리를 더 사용합니다.
    """

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "{repo_id}",
        trust_remote_code=True
    )

    # 기본 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(model, "{repo_id}")

    if merge_adapter:
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def translate(
    model,
    tokenizer,
    korean_text: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> str:
    """한국어 마크다운을 영어로 번역

    Args:
        model: 로드된 모델
        tokenizer: 토크나이저
        korean_text: 번역할 한국어 텍스트
        max_new_tokens: 최대 생성 토큰 수
        temperature: 샘플링 온도
        top_p: Top-p 샘플링

    Returns:
        번역된 영어 텍스트
    """
    system_prompt = (
        "You are an expert Korean to English translator specializing in technical documentation. "
        "Translate the given Korean markdown text to English while preserving all markdown formatting, "
        "including headers, lists, code blocks, links, and other structural elements."
    )

    user_prompt = f"Translate the following Korean markdown to English:\\n\\n{{korean_text}}"

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": user_prompt}}
    ]

    # 프롬프트 생성
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 토큰화
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 디코딩 (입력 제외)
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    translation = tokenizer.decode(generated, skip_special_tokens=True)

    return translation.strip()


if __name__ == "__main__":
    # 예제 사용법
    print("Loading model...")
    model, tokenizer = load_model(merge_adapter=False)

    # 예제 텍스트
    korean_text = """# 프로젝트 소개

이 프로젝트는 마크다운 문서를 번역하는 도구입니다.

## 기능

- 한국어에서 영어로 번역
- 마크다운 구조 보존
- 코드 블록 유지

## 설치 방법

```bash
pip install -r requirements.txt
```

자세한 내용은 [문서](./docs)를 참고하세요.
"""

    print("Translating...")
    translation = translate(model, tokenizer, korean_text)
    print("\\nTranslation:")
    print(translation)
'''

        example_path = output_dir / "inference_example.py"
        with open(example_path, 'w', encoding='utf-8') as f:
            f.write(example_code)

        print(f"Created inference example: {example_path}")

    def prepare_upload_directory(
        self,
        adapter_path: str,
        output_dir: Optional[str] = None,
        repo_id: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> Path:
        """업로드용 디렉토리 준비

        Args:
            adapter_path: 학습된 어댑터 경로
            output_dir: 출력 디렉토리 (없으면 임시 디렉토리)
            repo_id: 리포지토리 ID
            training_info: 학습 정보
            metrics: 평가 메트릭

        Returns:
            준비된 디렉토리 경로
        """
        adapter_path = Path(adapter_path)
        upload_config = self.config.get('upload', {})

        # 출력 디렉토리 설정
        if output_dir:
            upload_dir = Path(output_dir)
        else:
            upload_dir = adapter_path.parent / "hub_upload"

        upload_dir.mkdir(parents=True, exist_ok=True)

        # 어댑터 파일 복사
        adapter_files = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
        ]

        for file_name in adapter_files:
            src = adapter_path / file_name
            if src.exists():
                shutil.copy2(src, upload_dir / file_name)

        # 토크나이저 파일 복사
        if upload_config.get('include_tokenizer', True):
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            ]
            for file_name in tokenizer_files:
                src = adapter_path / file_name
                if src.exists():
                    shutil.copy2(src, upload_dir / file_name)

        # 설정 파일 복사
        if upload_config.get('include_config', True):
            config_files = ["config.json", "generation_config.json"]
            for file_name in config_files:
                src = adapter_path / file_name
                if src.exists():
                    shutil.copy2(src, upload_dir / file_name)

        # 모델 카드 생성
        if repo_id:
            self.create_model_card(repo_id, upload_dir, training_info, metrics)

        # 추론 예제 생성
        if upload_config.get('include_inference_example', True) and repo_id:
            self._create_inference_example(repo_id, upload_dir)

        # 업로드 메타데이터 저장
        metadata = {
            "upload_time": datetime.now().isoformat(),
            "source_path": str(adapter_path),
            "repo_id": repo_id,
            "training_info": training_info,
            "metrics": metrics
        }
        with open(upload_dir / ".upload_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return upload_dir

    def upload(
        self,
        adapter_path: str,
        repo_id: Optional[str] = None,
        private: Optional[bool] = None,
        commit_message: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        create_pr: bool = False
    ) -> str:
        """모델을 Hugging Face Hub에 업로드

        Args:
            adapter_path: 학습된 어댑터 경로
            repo_id: 리포지토리 ID (예: "username/model-name")
            private: 비공개 여부
            commit_message: 커밋 메시지
            training_info: 학습 정보
            metrics: 평가 메트릭
            create_pr: PR 생성 여부 (기존 리포지토리에 업데이트시)

        Returns:
            업로드된 리포지토리 URL
        """
        self._ensure_authenticated()

        hub_config = self.config.get('hub', {})

        # 리포지토리 ID
        repo_id = repo_id or hub_config.get('repo_id')
        if not repo_id:
            raise ValueError(
                "repo_id is required. Specify via argument, config file, or HUB_REPO_ID env var."
            )

        # 비공개 여부
        if private is None:
            private = hub_config.get('private', False)

        # 커밋 메시지
        commit_message = commit_message or hub_config.get(
            'commit_message',
            "Upload fine-tuned markdown translator model"
        )

        # 리포지토리 생성 또는 확인
        if hub_config.get('create_repo', True):
            try:
                create_repo(
                    repo_id=repo_id,
                    token=self.token,
                    private=private,
                    repo_type="model",
                    exist_ok=True
                )
                print(f"Repository ready: {repo_id}")
            except Exception as e:
                print(f"Repository creation/check: {e}")

        # 업로드 디렉토리 준비
        print("Preparing upload directory...")
        upload_dir = self.prepare_upload_directory(
            adapter_path=adapter_path,
            repo_id=repo_id,
            training_info=training_info,
            metrics=metrics
        )

        # 업로드
        print(f"Uploading to {repo_id}...")
        upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            token=self.token,
            commit_message=commit_message,
            create_pr=create_pr
        )

        repo_url = f"https://huggingface.co/{repo_id}"
        print(f"\nUpload complete!")
        print(f"Model available at: {repo_url}")

        return repo_url

    def upload_merged_model(
        self,
        base_model_name: str,
        adapter_path: str,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
        torch_dtype: str = "bfloat16"
    ) -> str:
        """병합된 전체 모델 업로드

        LoRA 어댑터를 기본 모델에 병합하여 단독으로 사용 가능한 모델을 업로드합니다.

        Args:
            base_model_name: 기본 모델 이름
            adapter_path: 어댑터 경로
            repo_id: 리포지토리 ID
            private: 비공개 여부
            commit_message: 커밋 메시지
            torch_dtype: 데이터 타입

        Returns:
            업로드된 리포지토리 URL
        """
        self._ensure_authenticated()

        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel

        print(f"Loading base model: {base_model_name}")

        # dtype 매핑
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # 기본 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )

        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        # 어댑터 로드 및 병합
        print(f"Loading and merging adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

        # 리포지토리 생성
        create_repo(
            repo_id=repo_id,
            token=self.token,
            private=private,
            repo_type="model",
            exist_ok=True
        )

        # 모델 및 토크나이저 업로드
        commit_message = commit_message or "Upload merged markdown translator model"

        print(f"Uploading merged model to {repo_id}...")
        model.push_to_hub(
            repo_id,
            token=self.token,
            commit_message=commit_message
        )
        tokenizer.push_to_hub(
            repo_id,
            token=self.token,
            commit_message=commit_message
        )

        # 모델 카드 생성 및 업로드
        card_config = self.config.get('model_card', {})
        card_config['description'] = card_config.get('description', '').replace(
            "LoRA adapter",
            "merged model"
        )

        repo_url = f"https://huggingface.co/{repo_id}"
        print(f"\nMerged model upload complete!")
        print(f"Model available at: {repo_url}")

        return repo_url


def quick_upload(
    adapter_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False
) -> str:
    """간편 업로드 함수

    Args:
        adapter_path: 어댑터 경로
        repo_id: 리포지토리 ID
        token: Hugging Face 토큰
        private: 비공개 여부

    Returns:
        리포지토리 URL
    """
    uploader = HubUploader(token=token)
    return uploader.upload(
        adapter_path=adapter_path,
        repo_id=repo_id,
        private=private
    )
