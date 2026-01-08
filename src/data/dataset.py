"""PyTorch Dataset 클래스

EXAONE 모델 학습을 위한 데이터셋 구현입니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """마크다운 번역 데이터셋

    JSONL 형식의 학습 데이터를 로드하여 EXAONE 모델 학습에 사용합니다.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 4096,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Args:
            data_path: JSONL 데이터 파일 경로
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
            padding: 패딩 전략
            truncation: 잘라내기 여부
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.data = self._load_data(data_path)

    def _load_data(self, path: Union[str, Path]) -> List[Dict]:
        """JSONL 파일 로드"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """메시지를 EXAONE 채팅 형식으로 변환

        EXAONE 3.5는 apply_chat_template을 사용합니다.
        """
        # apply_chat_template 사용
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return formatted

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item['messages']

        # 대화 형식으로 변환
        formatted_text = self._format_conversation(messages)

        # 토큰화
        encodings = self.tokenizer(
            formatted_text,
            truncation=self.truncation,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # Causal LM의 경우 labels = input_ids
        labels = input_ids.clone()

        # 패딩 토큰은 loss 계산에서 제외
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class TranslationDatasetWithMasking(Dataset):
    """프롬프트 마스킹이 적용된 번역 데이터셋

    시스템 프롬프트와 사용자 입력에 대해서는 loss를 계산하지 않고,
    어시스턴트 응답(번역)에 대해서만 loss를 계산합니다.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 4096,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Args:
            data_path: JSONL 데이터 파일 경로
            tokenizer: HuggingFace 토크나이저
            max_length: 최대 시퀀스 길이
            padding: 패딩 전략
            truncation: 잘라내기 여부
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.data = self._load_data(data_path)

        # 어시스턴트 응답 시작 토큰 확인
        self._setup_special_tokens()

    def _load_data(self, path: Union[str, Path]) -> List[Dict]:
        """JSONL 파일 로드"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _setup_special_tokens(self):
        """EXAONE 특수 토큰 설정"""
        # EXAONE 3.5의 어시스턴트 시작 토큰
        self.assistant_token = "[|assistant|]"
        self.endofturn_token = "[|endofturn|]"

        # 토큰 ID 확인 (토크나이저에서)
        self.assistant_token_ids = self.tokenizer.encode(
            self.assistant_token, add_special_tokens=False
        )

    def _create_labels_with_masking(
        self,
        input_ids: torch.Tensor,
        formatted_text: str
    ) -> torch.Tensor:
        """프롬프트 부분을 마스킹한 labels 생성

        어시스턴트 응답 부분만 loss 계산에 사용합니다.
        """
        labels = input_ids.clone()

        # 어시스턴트 응답 시작 위치 찾기
        text_before_assistant = formatted_text.split(self.assistant_token)[0] + self.assistant_token

        # 프롬프트 부분의 토큰 수 계산
        prompt_tokens = self.tokenizer.encode(
            text_before_assistant, add_special_tokens=False
        )
        prompt_length = len(prompt_tokens)

        # 프롬프트 부분 마스킹 (-100은 loss 계산에서 제외)
        labels[:prompt_length] = -100

        # 패딩 토큰도 마스킹
        labels[labels == self.tokenizer.pad_token_id] = -100

        return labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        messages = item['messages']

        # 대화 형식으로 변환
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 토큰화
        encodings = self.tokenizer(
            formatted_text,
            truncation=self.truncation,
            max_length=self.max_length,
            padding=self.padding,
            return_tensors="pt"
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)

        # 프롬프트 마스킹된 labels 생성
        labels = self._create_labels_with_masking(input_ids, formatted_text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class DataCollatorForTranslation:
    """번역 데이터 콜레이터

    배치 처리를 위한 데이터 콜레이터입니다.
    """

    def __init__(self, tokenizer, padding: str = "longest", max_length: int = None):
        """
        Args:
            tokenizer: HuggingFace 토크나이저
            padding: 패딩 전략 ("longest", "max_length")
            max_length: 최대 길이 (padding="max_length"일 때 사용)
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """배치 생성"""
        # 각 필드별로 텐서 수집
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        return batch


def create_dataloaders(
    train_path: Union[str, Path],
    valid_path: Union[str, Path],
    tokenizer,
    batch_size: int = 2,
    max_length: int = 4096,
    num_workers: int = 4,
    use_masking: bool = True
):
    """데이터로더 생성 유틸리티

    Args:
        train_path: 학습 데이터 경로
        valid_path: 검증 데이터 경로
        tokenizer: 토크나이저
        batch_size: 배치 크기
        max_length: 최대 시퀀스 길이
        num_workers: 워커 수
        use_masking: 프롬프트 마스킹 사용 여부

    Returns:
        (train_dataloader, valid_dataloader) 튜플
    """
    from torch.utils.data import DataLoader

    DatasetClass = TranslationDatasetWithMasking if use_masking else TranslationDataset

    train_dataset = DatasetClass(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    valid_dataset = DatasetClass(
        data_path=valid_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    collator = DataCollatorForTranslation(tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader
