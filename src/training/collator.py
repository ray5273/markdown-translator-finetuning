"""데이터 콜레이터 모듈

배치 처리를 위한 데이터 콜레이터를 정의합니다.
"""

from typing import Dict, List, Any
import torch
from transformers import PreTrainedTokenizer
from dataclasses import dataclass


@dataclass
class DataCollatorForCausalLM:
    """Causal LM 데이터 콜레이터

    패딩과 레이블 마스킹을 처리합니다.
    """

    tokenizer: PreTrainedTokenizer
    padding: str = "longest"
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 생성"""
        # input_ids, attention_mask, labels 추출
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # 이미 텐서인 경우
        if isinstance(input_ids[0], torch.Tensor):
            batch = {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attention_mask),
                "labels": torch.stack(labels)
            }
        else:
            # 리스트인 경우 패딩 처리
            batch = self._pad_sequences(input_ids, attention_mask, labels)

        return batch

    def _pad_sequences(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        labels: List[List[int]]
    ) -> Dict[str, torch.Tensor]:
        """시퀀스 패딩"""
        # 최대 길이 결정
        if self.padding == "longest":
            max_len = max(len(ids) for ids in input_ids)
        elif self.padding == "max_length" and self.max_length:
            max_len = self.max_length
        else:
            max_len = max(len(ids) for ids in input_ids)

        # 패딩 적용
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        pad_token_id = self.tokenizer.pad_token_id

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            padding_length = max_len - len(ids)

            # Right padding
            padded_input_ids.append(ids + [pad_token_id] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)
            padded_labels.append(lab + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


@dataclass
class DataCollatorForCompletionOnly:
    """완성 부분만 학습하는 데이터 콜레이터

    프롬프트 부분은 마스킹하고 응답 부분만 학습합니다.
    """

    tokenizer: PreTrainedTokenizer
    response_template: str = "[|assistant|]"
    padding: str = "longest"
    max_length: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 생성"""
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # 이미 텐서인 경우
        if isinstance(input_ids[0], torch.Tensor):
            batch_input_ids = torch.stack(input_ids)
            batch_attention_mask = torch.stack(attention_mask)
        else:
            # 패딩 처리
            max_len = max(len(ids) for ids in input_ids)
            pad_token_id = self.tokenizer.pad_token_id

            padded_input_ids = []
            padded_attention_mask = []

            for ids, mask in zip(input_ids, attention_mask):
                padding_length = max_len - len(ids)
                padded_input_ids.append(ids + [pad_token_id] * padding_length)
                padded_attention_mask.append(mask + [0] * padding_length)

            batch_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            batch_attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)

        # Labels 생성 (프롬프트 부분 마스킹)
        batch_labels = self._create_labels(batch_input_ids)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }

    def _create_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """레이블 생성 (프롬프트 마스킹)"""
        labels = input_ids.clone()

        # 응답 템플릿 토큰 ID
        response_token_ids = self.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )

        for i, ids in enumerate(input_ids):
            # 응답 시작 위치 찾기
            response_start = self._find_response_start(ids, response_token_ids)

            if response_start is not None:
                # 프롬프트 부분 마스킹
                labels[i, :response_start] = -100
            else:
                # 응답 템플릿을 찾지 못한 경우 전체 유지
                pass

            # 패딩 토큰 마스킹
            labels[i][labels[i] == self.tokenizer.pad_token_id] = -100

        return labels

    def _find_response_start(
        self,
        input_ids: torch.Tensor,
        response_token_ids: List[int]
    ) -> int:
        """응답 시작 위치 찾기"""
        response_len = len(response_token_ids)
        input_list = input_ids.tolist()

        for i in range(len(input_list) - response_len + 1):
            if input_list[i:i + response_len] == response_token_ids:
                return i + response_len

        return None
