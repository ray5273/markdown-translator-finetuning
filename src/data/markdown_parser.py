"""마크다운 파싱 및 구조 보존 모듈

번역 시 마크다운 구문 요소(코드 블록, 링크, 이미지 등)를 보존하기 위해
플레이스홀더로 치환하고 번역 후 복원하는 기능을 제공합니다.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SegmentType(Enum):
    """마크다운 세그먼트 타입"""
    TEXT = "text"
    CODE_BLOCK = "code_block"
    INLINE_CODE = "inline_code"
    LINK = "link"
    IMAGE = "image"
    HTML_TAG = "html_tag"
    HEADER = "header"
    TABLE = "table"
    LIST_ITEM = "list_item"


@dataclass
class MarkdownSegment:
    """마크다운 세그먼트"""
    content: str
    segment_type: SegmentType
    translatable: bool = True
    placeholder: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class MarkdownPreserver:
    """마크다운 구조 보존 처리기

    번역 시 코드 블록, URL 등 번역하지 않아야 할 요소를
    플레이스홀더로 치환하고, 번역 후 원본으로 복원합니다.
    """

    # 번역하지 않을 패턴들 (순서 중요 - 더 긴 패턴 먼저)
    PATTERNS = {
        # 코드 블록 (```language\n...\n```)
        'code_block': re.compile(r'```[\w]*\n[\s\S]*?```', re.MULTILINE),

        # 인라인 코드 (`code`)
        'inline_code': re.compile(r'`[^`\n]+`'),

        # 이미지 (![alt](url))
        'image': re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),

        # 링크 ([text](url)) - 텍스트는 번역, URL은 보존
        'link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),

        # HTML 태그
        'html_tag': re.compile(r'<[^>]+>'),

        # HTML 주석
        'html_comment': re.compile(r'<!--[\s\S]*?-->'),

        # 수학 수식 ($$...$$)
        'math_block': re.compile(r'\$\$[\s\S]*?\$\$'),

        # 인라인 수식 ($...$)
        'math_inline': re.compile(r'\$[^$\n]+\$'),
    }

    # 테이블 패턴 (별도 처리)
    TABLE_ROW_PATTERN = re.compile(r'^\|(.+)\|$', re.MULTILINE)
    TABLE_SEPARATOR_PATTERN = re.compile(r'^\|[\s\-:|]+\|$', re.MULTILINE)

    def __init__(self):
        self.reset()

    def reset(self):
        """내부 상태 초기화"""
        self.placeholder_map: Dict[str, str] = {}
        self.counter: int = 0

    def _generate_placeholder(self, prefix: str) -> str:
        """고유 플레이스홀더 생성"""
        placeholder = f"{{{{__{prefix}_{self.counter}__}}}}"
        self.counter += 1
        return placeholder

    def extract_and_replace(self, text: str) -> Tuple[str, Dict[str, str]]:
        """번역 불필요 요소를 플레이스홀더로 치환

        Args:
            text: 원본 마크다운 텍스트

        Returns:
            (치환된 텍스트, 플레이스홀더 맵)
        """
        self.reset()
        result = text

        # 1. 코드 블록 처리 (가장 먼저 - 내부에 다른 패턴 포함 가능)
        def replace_code_block(match):
            placeholder = self._generate_placeholder("CODE_BLOCK")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['code_block'].sub(replace_code_block, result)

        # 2. HTML 주석 처리
        def replace_html_comment(match):
            placeholder = self._generate_placeholder("HTML_COMMENT")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['html_comment'].sub(replace_html_comment, result)

        # 3. 수학 수식 처리 (블록)
        def replace_math_block(match):
            placeholder = self._generate_placeholder("MATH_BLOCK")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['math_block'].sub(replace_math_block, result)

        # 4. 수학 수식 처리 (인라인)
        def replace_math_inline(match):
            placeholder = self._generate_placeholder("MATH_INLINE")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['math_inline'].sub(replace_math_inline, result)

        # 5. 인라인 코드 처리
        def replace_inline_code(match):
            placeholder = self._generate_placeholder("INLINE_CODE")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['inline_code'].sub(replace_inline_code, result)

        # 6. 이미지 처리 (전체 보존)
        def replace_image(match):
            placeholder = self._generate_placeholder("IMAGE")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['image'].sub(replace_image, result)

        # 7. 링크 처리 (URL만 보존, 텍스트는 번역 대상)
        def replace_link(match):
            link_text = match.group(1)
            url = match.group(2)
            url_placeholder = self._generate_placeholder("URL")
            self.placeholder_map[url_placeholder] = url
            return f"[{link_text}]({url_placeholder})"

        result = self.PATTERNS['link'].sub(replace_link, result)

        # 8. HTML 태그 처리
        def replace_html_tag(match):
            placeholder = self._generate_placeholder("HTML_TAG")
            self.placeholder_map[placeholder] = match.group(0)
            return placeholder

        result = self.PATTERNS['html_tag'].sub(replace_html_tag, result)

        return result, self.placeholder_map.copy()

    def restore_placeholders(
        self,
        translated_text: str,
        placeholder_map: Dict[str, str]
    ) -> str:
        """번역 후 플레이스홀더를 원본으로 복원

        Args:
            translated_text: 번역된 텍스트 (플레이스홀더 포함)
            placeholder_map: 플레이스홀더 -> 원본 매핑

        Returns:
            복원된 텍스트
        """
        result = translated_text

        # 플레이스홀더를 원본으로 치환
        for placeholder, original in placeholder_map.items():
            result = result.replace(placeholder, original)

        return result

    def analyze_markdown(self, text: str) -> Dict[str, int]:
        """마크다운 요소 분석

        Args:
            text: 분석할 마크다운 텍스트

        Returns:
            각 요소 타입별 개수
        """
        counts = {
            'code_blocks': len(self.PATTERNS['code_block'].findall(text)),
            'inline_codes': len(self.PATTERNS['inline_code'].findall(text)),
            'images': len(self.PATTERNS['image'].findall(text)),
            'links': len(self.PATTERNS['link'].findall(text)),
            'html_tags': len(self.PATTERNS['html_tag'].findall(text)),
            'headers': len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE)),
            'tables': self._count_tables(text),
            'lists': len(re.findall(r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+', text, re.MULTILINE)),
            'bold': len(re.findall(r'\*\*[^*]+\*\*', text)),
            'italic': len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', text)),
        }
        return counts

    def _count_tables(self, text: str) -> int:
        """테이블 개수 계산"""
        # 테이블 구분선 개수로 테이블 수 추정
        separators = self.TABLE_SEPARATOR_PATTERN.findall(text)
        return len(separators)

    def validate_preservation(
        self,
        source: str,
        translated: str
    ) -> Tuple[bool, Dict[str, any]]:
        """마크다운 구조 보존 검증

        Args:
            source: 원본 텍스트
            translated: 번역된 텍스트

        Returns:
            (보존 여부, 상세 정보)
        """
        source_counts = self.analyze_markdown(source)
        translated_counts = self.analyze_markdown(translated)

        issues = []
        preservation_rates = {}

        for element, src_count in source_counts.items():
            trans_count = translated_counts.get(element, 0)

            if src_count > 0:
                rate = trans_count / src_count
                preservation_rates[element] = rate

                if trans_count < src_count:
                    issues.append(f"{element}: {src_count} -> {trans_count} (missing)")
                elif trans_count > src_count:
                    issues.append(f"{element}: {src_count} -> {trans_count} (extra)")
            else:
                preservation_rates[element] = 1.0

        is_valid = len(issues) == 0

        return is_valid, {
            'is_valid': is_valid,
            'preservation_rates': preservation_rates,
            'issues': issues,
            'source_counts': source_counts,
            'translated_counts': translated_counts
        }


class MarkdownChunker:
    """긴 마크다운 문서를 청크로 분할

    토큰 제한을 고려하여 마크다운 구조를 유지하면서 분할합니다.
    """

    def __init__(self, max_chunk_size: int = 2000):
        """
        Args:
            max_chunk_size: 최대 청크 크기 (문자 수)
        """
        self.max_chunk_size = max_chunk_size

    def chunk_by_headers(self, text: str) -> List[str]:
        """헤더 기준으로 청크 분할

        Args:
            text: 마크다운 텍스트

        Returns:
            청크 리스트
        """
        # 헤더로 분할
        header_pattern = re.compile(r'^(#{1,6}\s+.+)$', re.MULTILINE)

        chunks = []
        current_chunk = []
        current_size = 0

        lines = text.split('\n')

        for line in lines:
            line_size = len(line) + 1  # +1 for newline

            # 헤더를 만나면 새 청크 시작 (크기 초과 시)
            if header_pattern.match(line) and current_size + line_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

                # 청크 크기 초과 시 분할
                if current_size > self.max_chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

        # 마지막 청크 추가
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """문단 기준으로 청크 분할

        Args:
            text: 마크다운 텍스트

        Returns:
            청크 리스트
        """
        # 빈 줄로 문단 분할
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para) + 2  # +2 for double newline

            if current_size + para_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks
