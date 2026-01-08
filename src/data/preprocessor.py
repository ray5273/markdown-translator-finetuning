"""데이터 전처리 파이프라인

한영 병렬 데이터를 EXAONE 학습 형식으로 변환합니다.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, asdict
from tqdm import tqdm

# 상대/절대 import 모두 지원
try:
    from .markdown_parser import MarkdownPreserver
except ImportError:
    from markdown_parser import MarkdownPreserver


@dataclass
class TranslationExample:
    """번역 학습 예시"""
    korean: str
    english: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessedExample:
    """전처리된 학습 예시"""
    messages: List[Dict[str, str]]
    metadata: Dict


class DataPreprocessor:
    """학습 데이터 전처리기

    한영 병렬 데이터를 EXAONE 채팅 형식으로 변환합니다.
    """

    # 시스템 프롬프트
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

    # 사용자 프롬프트 템플릿 (다양성을 위해 여러 버전)
    USER_TEMPLATES = [
        "Translate the following Korean markdown content to English. Preserve all markdown formatting exactly.\n\n{korean_text}",
        "Please translate this Korean document to English while keeping the markdown structure intact:\n\n{korean_text}",
        "Convert the following Korean text to English. Maintain all markdown formatting:\n\n{korean_text}",
        "Translate to English (preserve markdown formatting):\n\n{korean_text}",
    ]

    def __init__(
        self,
        max_length: int = 4096,
        min_length: int = 10,
        use_template_variety: bool = True,
        seed: int = 42
    ):
        """
        Args:
            max_length: 최대 텍스트 길이 (문자)
            min_length: 최소 텍스트 길이 (문자)
            use_template_variety: 다양한 프롬프트 템플릿 사용 여부
            seed: 랜덤 시드
        """
        self.max_length = max_length
        self.min_length = min_length
        self.use_template_variety = use_template_variety
        self.markdown_preserver = MarkdownPreserver()
        random.seed(seed)

    def _get_user_prompt(self, korean_text: str) -> str:
        """사용자 프롬프트 생성"""
        if self.use_template_variety:
            template = random.choice(self.USER_TEMPLATES)
        else:
            template = self.USER_TEMPLATES[0]
        return template.format(korean_text=korean_text)

    def _analyze_content(self, text: str) -> Dict:
        """콘텐츠 분석 (메타데이터 생성)"""
        md_counts = self.markdown_preserver.analyze_markdown(text)

        return {
            'char_count': len(text),
            'line_count': text.count('\n') + 1,
            'has_code_block': md_counts['code_blocks'] > 0,
            'has_table': md_counts['tables'] > 0,
            'has_link': md_counts['links'] > 0,
            'has_image': md_counts['images'] > 0,
            'markdown_elements': md_counts
        }

    def create_training_example(
        self,
        korean_text: str,
        english_text: str,
        additional_metadata: Dict = None
    ) -> Optional[ProcessedExample]:
        """학습 예시 생성

        Args:
            korean_text: 한국어 텍스트
            english_text: 영어 번역
            additional_metadata: 추가 메타데이터

        Returns:
            ProcessedExample 또는 None (필터링된 경우)
        """
        # 길이 필터링
        if len(korean_text) < self.min_length or len(english_text) < self.min_length:
            return None

        if len(korean_text) > self.max_length or len(english_text) > self.max_length:
            return None

        # 메시지 구성
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": self._get_user_prompt(korean_text)
            },
            {
                "role": "assistant",
                "content": english_text
            }
        ]

        # 메타데이터 생성
        metadata = self._analyze_content(korean_text)
        if additional_metadata:
            metadata.update(additional_metadata)

        return ProcessedExample(messages=messages, metadata=metadata)

    def process_parallel_corpus(
        self,
        data: List[TranslationExample],
        show_progress: bool = True
    ) -> List[ProcessedExample]:
        """병렬 코퍼스 전처리

        Args:
            data: 번역 예시 리스트
            show_progress: 진행률 표시 여부

        Returns:
            전처리된 예시 리스트
        """
        processed = []
        iterator = tqdm(data, desc="Processing") if show_progress else data

        for example in iterator:
            result = self.create_training_example(
                korean_text=example.korean,
                english_text=example.english,
                additional_metadata=example.metadata
            )
            if result:
                processed.append(result)

        return processed

    def split_dataset(
        self,
        data: List[ProcessedExample],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42
    ) -> Tuple[List[ProcessedExample], List[ProcessedExample], List[ProcessedExample]]:
        """데이터셋 분할

        Args:
            data: 전체 데이터
            train_ratio: 학습 데이터 비율
            valid_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            shuffle: 셔플 여부
            seed: 랜덤 시드

        Returns:
            (train, valid, test) 튜플
        """
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

        if shuffle:
            random.seed(seed)
            data = data.copy()
            random.shuffle(data)

        n = len(data)
        train_end = int(n * train_ratio)
        valid_end = train_end + int(n * valid_ratio)

        train_data = data[:train_end]
        valid_data = data[train_end:valid_end]
        test_data = data[valid_end:]

        return train_data, valid_data, test_data

    def save_jsonl(
        self,
        data: List[ProcessedExample],
        output_path: Union[str, Path]
    ):
        """JSONL 형식으로 저장

        Args:
            data: 저장할 데이터
            output_path: 출력 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in data:
                line = json.dumps({
                    'messages': example.messages,
                    'metadata': example.metadata
                }, ensure_ascii=False)
                f.write(line + '\n')

    def load_jsonl(
        self,
        input_path: Union[str, Path]
    ) -> List[ProcessedExample]:
        """JSONL 형식에서 로드

        Args:
            input_path: 입력 경로

        Returns:
            ProcessedExample 리스트
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(ProcessedExample(
                    messages=item['messages'],
                    metadata=item.get('metadata', {})
                ))
        return data

    def process_and_save(
        self,
        data: List[TranslationExample],
        output_dir: Union[str, Path],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, int]:
        """전처리 및 저장 파이프라인

        Args:
            data: 원본 데이터
            output_dir: 출력 디렉토리
            train_ratio: 학습 데이터 비율
            valid_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율

        Returns:
            각 분할의 데이터 수
        """
        output_dir = Path(output_dir)

        # 전처리
        processed = self.process_parallel_corpus(data)
        print(f"Processed {len(processed)} examples from {len(data)} raw examples")

        # 분할
        train, valid, test = self.split_dataset(
            processed,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )

        # 저장
        self.save_jsonl(train, output_dir / 'train.jsonl')
        self.save_jsonl(valid, output_dir / 'valid.jsonl')
        self.save_jsonl(test, output_dir / 'test.jsonl')

        stats = {
            'total': len(processed),
            'train': len(train),
            'valid': len(valid),
            'test': len(test)
        }

        print(f"Saved: train={stats['train']}, valid={stats['valid']}, test={stats['test']}")
        return stats


class SyntheticDataGenerator:
    """합성 데이터 생성기

    LLM API를 사용하여 한영 번역 쌍을 생성합니다.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
        """
        self.api_key = api_key
        self.model = model

    def generate_translation_pair(
        self,
        korean_text: str
    ) -> Optional[str]:
        """한국어 텍스트에 대한 영어 번역 생성

        Args:
            korean_text: 번역할 한국어 텍스트

        Returns:
            영어 번역 또는 None
        """
        # TODO: OpenAI API 호출 구현
        # 현재는 플레이스홀더
        raise NotImplementedError("API integration required")

    def generate_markdown_content(
        self,
        topic: str,
        style: str = "technical"
    ) -> Optional[Tuple[str, str]]:
        """주제에 대한 마크다운 콘텐츠 생성 (한영 동시)

        Args:
            topic: 콘텐츠 주제
            style: 스타일 (technical, tutorial, readme 등)

        Returns:
            (한국어, 영어) 튜플 또는 None
        """
        # TODO: API 호출로 양방향 생성
        raise NotImplementedError("API integration required")
