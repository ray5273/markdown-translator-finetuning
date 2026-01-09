"""합성 데이터 생성 모듈

LLM API를 사용하여 한영 마크다운 번역 쌍을 생성합니다.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from tqdm import tqdm

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Local LLM providers
LOCAL_LLM_AVAILABLE = False
try:
    from .local_llm_provider import (
        OllamaProvider as _OllamaProvider,
        VLLMProvider as _VLLMProvider,
        check_ollama_available,
        check_vllm_available,
    )
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    try:
        # Fallback for dynamic module loading (when run via load_module)
        from pathlib import Path
        import importlib.util
        _provider_path = Path(__file__).parent / "local_llm_provider.py"
        if _provider_path.exists():
            _spec = importlib.util.spec_from_file_location("local_llm_provider", _provider_path)
            _local_llm_module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_local_llm_module)
            _OllamaProvider = _local_llm_module.OllamaProvider
            _VLLMProvider = _local_llm_module.VLLMProvider
            check_ollama_available = _local_llm_module.check_ollama_available
            check_vllm_available = _local_llm_module.check_vllm_available
            LOCAL_LLM_AVAILABLE = True
    except Exception:
        pass


@dataclass
class GeneratedPair:
    """생성된 한영 번역 쌍"""
    korean: str
    english: str
    topic: str
    style: str
    model: str
    metadata: Dict = None


class LLMProvider(ABC):
    """LLM 프로바이더 추상 클래스"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API 프로바이더"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic API 프로바이더"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )

        return response.content[0].text


class OllamaProvider(LLMProvider):
    """Ollama API 프로바이더

    로컬에서 실행되는 Ollama 서버와 통신합니다.

    설치:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.1:8b
        ollama serve
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0
    ):
        if not LOCAL_LLM_AVAILABLE:
            raise ImportError(
                "Local LLM provider not available. "
                "Make sure requests package is installed: pip install requests"
            )

        self._provider = _OllamaProvider(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return self._provider.generate(prompt, system_prompt)


class VLLMProvider(LLMProvider):
    """vLLM OpenAI 호환 API 프로바이더

    vLLM 서버의 OpenAI 호환 API를 사용합니다.

    vLLM 서버 시작:
        python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Llama-3.1-8B-Instruct \\
            --port 8000
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str = "EMPTY"
    ):
        if not LOCAL_LLM_AVAILABLE:
            raise ImportError(
                "Local LLM provider not available. "
                "Make sure openai package is installed: pip install openai"
            )

        self._provider = _VLLMProvider(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
        self.model = self._provider.model

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        return self._provider.generate(prompt, system_prompt)


class SyntheticDataGenerator:
    """합성 데이터 생성기

    LLM을 사용하여 다양한 스타일의 한영 마크다운 번역 쌍을 생성합니다.
    """

    # 기술 문서 주제들
    TOPICS = [
        # 프로그래밍
        "Python 비동기 프로그래밍",
        "JavaScript 프레임워크 비교",
        "Rust 메모리 관리",
        "Go 동시성 패턴",
        "TypeScript 타입 시스템",
        "React 상태 관리",
        "Vue.js 컴포넌트 설계",
        "Node.js 성능 최적화",
        "Docker 컨테이너화",
        "Kubernetes 클러스터 관리",

        # 머신러닝/AI
        "딥러닝 모델 학습",
        "자연어 처리 기초",
        "트랜스포머 아키텍처",
        "모델 파인튜닝 가이드",
        "MLOps 파이프라인",
        "LLM 프롬프트 엔지니어링",

        # 데이터베이스
        "PostgreSQL 최적화",
        "MongoDB 스키마 설계",
        "Redis 캐싱 전략",
        "SQL 쿼리 튜닝",

        # DevOps/인프라
        "CI/CD 파이프라인 구축",
        "AWS 서버리스 아키텍처",
        "Terraform 인프라 코드",
        "Nginx 설정 가이드",

        # 기타
        "Git 브랜치 전략",
        "API 설계 원칙",
        "마이크로서비스 아키텍처",
        "보안 모범 사례",
        "성능 모니터링",
        "로깅 시스템 구축",
    ]

    # 문서 스타일
    STYLES = {
        "readme": "프로젝트 README 문서 스타일 (설치 방법, 사용법, 예제 포함)",
        "tutorial": "단계별 튜토리얼 스타일 (상세한 설명과 코드 예제)",
        "api_doc": "API 문서 스타일 (함수 시그니처, 파라미터, 반환값 설명)",
        "blog": "기술 블로그 스타일 (친근한 어조, 경험 공유)",
        "reference": "레퍼런스 문서 스타일 (간결하고 정확한 설명)",
        "troubleshooting": "문제 해결 가이드 스타일 (증상, 원인, 해결책)",
    }

    # 스타일별 필수 마크다운 요소
    STYLE_REQUIRED_ELEMENTS = {
        "readme": ["headers", "code_blocks", "lists", "links", "bold"],
        "tutorial": ["headers", "code_blocks", "inline_codes", "lists", "links", "blockquotes"],
        "api_doc": ["headers", "code_blocks", "inline_codes", "tables", "lists"],
        "blog": ["headers", "code_blocks", "links", "bold", "italic", "blockquotes"],
        "reference": ["headers", "tables", "inline_codes", "lists", "links"],
        "troubleshooting": ["headers", "code_blocks", "lists", "blockquotes", "bold"],
    }

    # 마크다운 요소별 설명 (프롬프트에서 사용)
    MARKDOWN_ELEMENT_DESCRIPTIONS = {
        "headers": "헤더 (# ## ###)",
        "code_blocks": "코드 블록 (```language ... ```)",
        "inline_codes": "인라인 코드 (`code`)",
        "lists": "리스트 (- 또는 1. 2. 3.)",
        "bold": "굵은 글씨 (**bold**)",
        "italic": "기울임 (*italic*)",
        "links": "링크 ([텍스트](URL))",
        "images": "이미지 (![alt text](image_url))",
        "tables": "테이블 (| col1 | col2 | 형식)",
        "blockquotes": "인용구 (> quote)",
        "math_inline": "인라인 수식 ($수식$)",
        "math_block": "블록 수식 ($$수식$$)",
    }

    SYSTEM_PROMPT = """당신은 기술 문서 전문 번역가이자 작가입니다.
한국어와 영어 모두 네이티브 수준으로 구사하며,
마크다운 형식의 기술 문서를 작성하는 데 전문성을 가지고 있습니다.

중요 규칙:
1. 마크다운 문법을 다양하고 적극적으로 활용하세요:
   - 헤더 (# ## ###)
   - 코드 블록 (```language ... ```)
   - 인라인 코드 (`code`)
   - 링크 ([텍스트](URL))
   - 이미지 (![alt](url)) - 적절한 경우에만
   - 테이블 (| col1 | col2 |)
   - 리스트 (순서 있는/없는)
   - 인용구 (> quote)
   - 굵은 글씨 (**bold**) 및 기울임 (*italic*)
2. 코드 예제는 실제로 동작하는 코드를 작성하세요
3. 기술 용어는 정확하게 사용하세요
4. 한국어 버전과 영어 버전의 구조와 마크다운 요소는 동일하게 유지하세요
5. 자연스럽고 읽기 쉬운 문서를 작성하세요
6. 테이블은 최소 2행 이상, 헤더와 구분선을 포함하세요
7. 링크는 실제 존재할 법한 URL 형식으로 작성하세요 (예: https://example.com/docs)"""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        **kwargs
    ):
        """
        Args:
            provider: LLM 프로바이더 ("openai", "anthropic", "ollama", "vllm" 또는 LLMProvider 인스턴스)
            api_key: API 키 (OpenAI, Anthropic용)
            model: 사용할 모델
            base_url: 서버 URL (Ollama, vLLM용)
            **kwargs: 추가 설정 (timeout, max_tokens 등)
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
            self.model_name = getattr(provider, 'model', 'custom')
        elif provider == "openai":
            self.provider = OpenAIProvider(
                api_key=api_key,
                model=model or "gpt-4o"
            )
            self.model_name = model or "gpt-4o"
        elif provider == "anthropic":
            self.provider = AnthropicProvider(
                api_key=api_key,
                model=model or "claude-sonnet-4-20250514"
            )
            self.model_name = model or "claude-sonnet-4-20250514"
        elif provider == "ollama":
            self.provider = OllamaProvider(
                base_url=base_url or "http://localhost:11434",
                model=model or "llama3.1:8b",
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
                timeout=kwargs.get("timeout", 120.0)
            )
            self.model_name = model or "llama3.1:8b"
        elif provider == "vllm":
            self.provider = VLLMProvider(
                base_url=base_url or "http://localhost:8000",
                model=model,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
                api_key=kwargs.get("api_key", "EMPTY")
            )
            self.model_name = self.provider.model
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, ollama, vllm")

    def _get_required_elements_prompt(self, style: str) -> str:
        """스타일별 필수 마크다운 요소 프롬프트 생성"""
        required = self.STYLE_REQUIRED_ELEMENTS.get(style, self.STYLE_REQUIRED_ELEMENTS["tutorial"])
        elements_desc = []
        for elem in required:
            if elem in self.MARKDOWN_ELEMENT_DESCRIPTIONS:
                elements_desc.append(f"   - {self.MARKDOWN_ELEMENT_DESCRIPTIONS[elem]}")
        return "\n".join(elements_desc)

    def generate_bilingual_content(
        self,
        topic: str,
        style: str = "tutorial"
    ) -> Optional[GeneratedPair]:
        """주제에 대한 한영 병렬 콘텐츠 생성

        Args:
            topic: 문서 주제
            style: 문서 스타일

        Returns:
            GeneratedPair 또는 None
        """
        style_desc = self.STYLES.get(style, self.STYLES["tutorial"])
        required_elements = self._get_required_elements_prompt(style)

        prompt = f"""다음 주제와 스타일로 마크다운 기술 문서를 작성해주세요.

주제: {topic}
스타일: {style_desc}

요구사항:
1. 먼저 한국어로 완전한 문서를 작성하세요
2. 그 다음 동일한 내용의 영어 번역을 작성하세요
3. 다음 마크다운 요소를 반드시 포함하세요 (이 스타일의 필수 요소):
{required_elements}
4. 추가로 다음 요소 중 2개 이상을 포함하면 좋습니다:
   - 테이블 (| 열1 | 열2 | 형식, 최소 헤더 + 구분선 + 2행)
   - 링크 ([텍스트](https://example.com/path))
   - 인용구 (> 중요한 내용)
   - 이미지 참조 (![설명](https://example.com/image.png))
5. 문서 길이: 400-1000 단어
6. 한국어와 영어 버전은 동일한 마크다운 구조를 유지하세요

테이블 예시 (반드시 이 형식을 따르세요):
| 항목 | 설명 |
|------|------|
| 값1 | 설명1 |
| 값2 | 설명2 |

출력 형식:
---KOREAN---
[한국어 마크다운 문서]
---ENGLISH---
[영어 마크다운 문서]
---END---"""

        try:
            response = self.provider.generate(prompt, self.SYSTEM_PROMPT)

            # 파싱
            if "---KOREAN---" in response and "---ENGLISH---" in response:
                korean_start = response.find("---KOREAN---") + len("---KOREAN---")
                korean_end = response.find("---ENGLISH---")
                english_start = korean_end + len("---ENGLISH---")
                english_end = response.find("---END---") if "---END---" in response else len(response)

                korean = response[korean_start:korean_end].strip()
                english = response[english_start:english_end].strip()

                if korean and english:
                    pair = GeneratedPair(
                        korean=korean,
                        english=english,
                        topic=topic,
                        style=style,
                        model=self.model_name
                    )
                    # 생성된 콘텐츠 검증
                    validation = self.validate_generated_content(pair, style)
                    pair.metadata = {"validation": validation}
                    return pair

        except Exception as e:
            print(f"Generation failed for topic '{topic}': {e}")

        return None

    def validate_generated_content(
        self,
        pair: GeneratedPair,
        style: str
    ) -> Dict:
        """생성된 콘텐츠의 마크다운 요소 검증

        Args:
            pair: 생성된 한영 쌍
            style: 문서 스타일

        Returns:
            검증 결과 딕셔너리
        """
        import re

        def count_elements(text: str) -> Dict[str, int]:
            """텍스트에서 마크다운 요소 개수 계산"""
            return {
                "headers": len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE)),
                "code_blocks": len(re.findall(r'```[\w]*\n[\s\S]*?```', text)),
                "inline_codes": len(re.findall(r'`[^`\n]+`', text)),
                "lists": len(re.findall(r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+', text, re.MULTILINE)),
                "bold": len(re.findall(r'\*\*[^*]+\*\*', text)),
                "italic": len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', text)),
                "links": len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)),
                "images": len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text)),
                "tables": len(re.findall(r'^\|[\s\-:|]+\|$', text, re.MULTILINE)),
                "blockquotes": len(re.findall(r'^>\s+', text, re.MULTILINE)),
            }

        korean_counts = count_elements(pair.korean)
        english_counts = count_elements(pair.english)

        required = self.STYLE_REQUIRED_ELEMENTS.get(style, [])
        missing_korean = []
        missing_english = []

        for elem in required:
            if korean_counts.get(elem, 0) == 0:
                missing_korean.append(elem)
            if english_counts.get(elem, 0) == 0:
                missing_english.append(elem)

        is_valid = len(missing_korean) == 0 and len(missing_english) == 0

        return {
            "is_valid": is_valid,
            "korean_counts": korean_counts,
            "english_counts": english_counts,
            "required_elements": required,
            "missing_korean": missing_korean,
            "missing_english": missing_english,
        }

    def generate_translation_pair(
        self,
        korean_text: str
    ) -> Optional[str]:
        """한국어 텍스트에 대한 영어 번역 생성

        Args:
            korean_text: 번역할 한국어 마크다운

        Returns:
            영어 번역
        """
        prompt = f"""다음 한국어 마크다운 문서를 영어로 번역해주세요.

요구사항:
1. 모든 마크다운 서식을 정확히 보존하세요
2. 코드 블록 내용은 번역하지 마세요
3. URL과 파일 경로는 그대로 유지하세요
4. 자연스럽고 유창한 영어로 번역하세요

한국어 문서:
{korean_text}

영어 번역:"""

        try:
            return self.provider.generate(prompt, self.SYSTEM_PROMPT)
        except Exception as e:
            print(f"Translation failed: {e}")
            return None

    def generate_dataset(
        self,
        num_samples: int = 100,
        output_path: str = "data/synthetic/generated_pairs.jsonl",
        topics: List[str] = None,
        styles: List[str] = None,
        delay: float = 1.0
    ) -> List[GeneratedPair]:
        """데이터셋 생성

        Args:
            num_samples: 생성할 샘플 수
            output_path: 저장 경로
            topics: 사용할 주제 리스트 (None이면 기본값)
            styles: 사용할 스타일 리스트 (None이면 기본값)
            delay: API 호출 간 딜레이 (초)

        Returns:
            생성된 쌍 리스트
        """
        topics = topics or self.TOPICS
        styles = styles or list(self.STYLES.keys())

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generated = []

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in tqdm(range(num_samples), desc="Generating samples"):
                topic = random.choice(topics)
                style = random.choice(styles)

                pair = self.generate_bilingual_content(topic, style)

                if pair:
                    generated.append(pair)

                    # 저장
                    data = {
                        "korean": pair.korean,
                        "english": pair.english,
                        "metadata": {
                            "topic": pair.topic,
                            "style": pair.style,
                            "model": pair.model,
                            "source": "synthetic"
                        }
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f.flush()

                time.sleep(delay)

        print(f"Generated {len(generated)} samples, saved to {output_path}")
        return generated


class TemplateBasedGenerator:
    """템플릿 기반 데이터 생성기

    API 없이 템플릿을 사용하여 간단한 예제 데이터를 생성합니다.
    """

    # 템플릿들 (한영 쌍)
    TEMPLATES = [
        # README 스타일
        {
            "korean": """# {project_name}

{description_ko}

## 설치 방법

```bash
pip install {package_name}
```

## 사용법

```python
from {package_name} import {class_name}

{class_name_lower} = {class_name}()
result = {class_name_lower}.run()
```

## 기능

- **{feature1_ko}**: {feature1_desc_ko}
- **{feature2_ko}**: {feature2_desc_ko}

## 라이선스

MIT License""",

            "english": """# {project_name}

{description_en}

## Installation

```bash
pip install {package_name}
```

## Usage

```python
from {package_name} import {class_name}

{class_name_lower} = {class_name}()
result = {class_name_lower}.run()
```

## Features

- **{feature1_en}**: {feature1_desc_en}
- **{feature2_en}**: {feature2_desc_en}

## License

MIT License"""
        },

        # 함수 문서 스타일
        {
            "korean": """## `{func_name}`

{func_desc_ko}

### 매개변수

| 이름 | 타입 | 설명 |
|------|------|------|
| `{param1}` | `{type1}` | {param1_desc_ko} |
| `{param2}` | `{type2}` | {param2_desc_ko} |

### 반환값

- `{return_type}`: {return_desc_ko}

### 예제

```python
result = {func_name}({param1}={param1_val}, {param2}={param2_val})
print(result)
```

### 참고

> {note_ko}""",

            "english": """## `{func_name}`

{func_desc_en}

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `{param1}` | `{type1}` | {param1_desc_en} |
| `{param2}` | `{type2}` | {param2_desc_en} |

### Returns

- `{return_type}`: {return_desc_en}

### Example

```python
result = {func_name}({param1}={param1_val}, {param2}={param2_val})
print(result)
```

### Note

> {note_en}"""
        },

        # 튜토리얼 스타일 (링크, 이미지, 인용구 포함)
        {
            "korean": """# {tutorial_title_ko}

{tutorial_intro_ko}

![다이어그램](https://example.com/images/{diagram_name}.png)

## 사전 준비

시작하기 전에 다음 항목을 확인하세요:

- **{prereq1_ko}**가 설치되어 있어야 합니다
- **{prereq2_ko}** 계정이 필요합니다

> **중요**: {important_note_ko}

## 1단계: 설치

먼저 `{package_name}`을 설치합니다:

```bash
pip install {package_name}
```

자세한 내용은 [공식 문서](https://example.com/docs/{package_name})를 참조하세요.

## 2단계: 설정

설정 파일을 생성합니다:

```python
# config.py
{config_code}
```

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `timeout` | `30` | {timeout_desc_ko} |
| `retries` | `3` | {retries_desc_ko} |
| `debug` | `False` | {debug_desc_ko} |

## 3단계: 실행

다음 명령어로 실행합니다:

```bash
python main.py --config config.py
```

## 관련 링크

- [GitHub 저장소](https://github.com/example/{package_name})
- [API 레퍼런스](https://example.com/api/{package_name})
- [이슈 트래커](https://github.com/example/{package_name}/issues)""",

            "english": """# {tutorial_title_en}

{tutorial_intro_en}

![Diagram](https://example.com/images/{diagram_name}.png)

## Prerequisites

Before you begin, make sure you have:

- **{prereq1_en}** installed
- A **{prereq2_en}** account

> **Important**: {important_note_en}

## Step 1: Installation

First, install `{package_name}`:

```bash
pip install {package_name}
```

For more details, refer to the [official documentation](https://example.com/docs/{package_name}).

## Step 2: Configuration

Create a configuration file:

```python
# config.py
{config_code}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `timeout` | `30` | {timeout_desc_en} |
| `retries` | `3` | {retries_desc_en} |
| `debug` | `False` | {debug_desc_en} |

## Step 3: Execution

Run with the following command:

```bash
python main.py --config config.py
```

## Related Links

- [GitHub Repository](https://github.com/example/{package_name})
- [API Reference](https://example.com/api/{package_name})
- [Issue Tracker](https://github.com/example/{package_name}/issues)"""
        },

        # 블로그 스타일 (이탤릭, 인용구, 링크 강조)
        {
            "korean": """# {blog_title_ko}

*{blog_date}* | 작성자: {author_name}

{blog_intro_ko}

## {section1_title_ko}

{section1_content_ko}

> "{quote_ko}"
> — *{quote_author}*

### 핵심 포인트

1. **{point1_ko}**: {point1_desc_ko}
2. **{point2_ko}**: {point2_desc_ko}
3. **{point3_ko}**: {point3_desc_ko}

## 코드 예제

다음은 간단한 구현 예제입니다:

```python
{blog_code}
```

출력 결과:

```
{code_output}
```

## 성능 비교

| 방법 | 속도 | 메모리 |
|------|------|--------|
| {method1} | {speed1} | {memory1} |
| {method2} | {speed2} | {memory2} |

## 결론

{conclusion_ko}

더 자세한 내용은 [여기](https://example.com/blog/{blog_slug})에서 확인하세요.

---

*이 글이 도움이 되었다면 [Twitter](https://twitter.com/example)에서 공유해주세요!*""",

            "english": """# {blog_title_en}

*{blog_date}* | Author: {author_name}

{blog_intro_en}

## {section1_title_en}

{section1_content_en}

> "{quote_en}"
> — *{quote_author}*

### Key Points

1. **{point1_en}**: {point1_desc_en}
2. **{point2_en}**: {point2_desc_en}
3. **{point3_en}**: {point3_desc_en}

## Code Example

Here's a simple implementation example:

```python
{blog_code}
```

Output:

```
{code_output}
```

## Performance Comparison

| Method | Speed | Memory |
|--------|-------|--------|
| {method1} | {speed1} | {memory1} |
| {method2} | {speed2} | {memory2} |

## Conclusion

{conclusion_en}

For more details, check out [this link](https://example.com/blog/{blog_slug}).

---

*If you found this helpful, please share on [Twitter](https://twitter.com/example)!*"""
        },

        # 문제 해결 가이드 스타일
        {
            "korean": """# 문제 해결: {error_name}

## 증상

{symptom_ko}

다음과 같은 에러 메시지가 나타납니다:

```
{error_message}
```

## 원인

이 문제는 주로 다음과 같은 원인으로 발생합니다:

- **{cause1_ko}**
- **{cause2_ko}**
- **{cause3_ko}**

> **참고**: {cause_note_ko}

## 해결 방법

### 방법 1: {solution1_title_ko}

```bash
{solution1_command}
```

### 방법 2: {solution2_title_ko}

설정 파일을 수정합니다:

```python
{solution2_code}
```

### 방법 3: 환경 변수 확인

| 환경 변수 | 예상 값 | 설명 |
|-----------|---------|------|
| `{env1}` | `{env1_val}` | {env1_desc_ko} |
| `{env2}` | `{env2_val}` | {env2_desc_ko} |

## 추가 리소스

- [관련 이슈](https://github.com/example/repo/issues/{issue_num})
- [공식 문서](https://example.com/docs/troubleshooting)

문제가 지속되면 [지원팀](https://example.com/support)에 문의하세요.""",

            "english": """# Troubleshooting: {error_name}

## Symptoms

{symptom_en}

You may see the following error message:

```
{error_message}
```

## Causes

This issue typically occurs due to:

- **{cause1_en}**
- **{cause2_en}**
- **{cause3_en}**

> **Note**: {cause_note_en}

## Solutions

### Method 1: {solution1_title_en}

```bash
{solution1_command}
```

### Method 2: {solution2_title_en}

Modify your configuration file:

```python
{solution2_code}
```

### Method 3: Check Environment Variables

| Variable | Expected Value | Description |
|----------|----------------|-------------|
| `{env1}` | `{env1_val}` | {env1_desc_en} |
| `{env2}` | `{env2_val}` | {env2_desc_en} |

## Additional Resources

- [Related Issue](https://github.com/example/repo/issues/{issue_num})
- [Official Documentation](https://example.com/docs/troubleshooting)

If the problem persists, contact [support](https://example.com/support)."""
        },
    ]

    # 치환 변수들
    VARIABLES = {
        "project_name": ["DataProcessor", "ImageAnalyzer", "TextParser", "APIClient", "FileManager"],
        "package_name": ["dataproc", "imganalyze", "textparse", "apiclient", "filemgr"],
        "class_name": ["Processor", "Analyzer", "Parser", "Client", "Manager"],
        "func_name": ["process_data", "analyze_image", "parse_text", "fetch_api", "read_file"],
        "param1": ["input", "data", "text", "path"],
        "type1": ["str", "bytes", "List[str]", "Path"],
        "param1_val": ['"hello"', 'b"data"', '["a", "b"]', 'Path("./file")'],
        "param2": ["options", "config", "format", "encoding"],
        "type2": ["dict", "Config", "str", "str"],
        "param2_val": ['{}', 'Config()', '"json"', '"utf-8"'],
        "return_type": ["Result", "dict", "List[str]", "bytes"],
        # 튜토리얼 템플릿 변수
        "diagram_name": ["architecture", "workflow", "dataflow", "pipeline"],
        "config_code": [
            "CONFIG = {\n    'timeout': 30,\n    'retries': 3,\n}",
            "settings = Settings(\n    debug=True,\n    log_level='INFO',\n)",
        ],
        # 블로그 템플릿 변수
        "blog_date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "author_name": ["김철수", "이영희", "박민수"],
        "blog_slug": ["performance-tips", "best-practices", "getting-started"],
        "blog_code": [
            "def optimize(data):\n    return sorted(data, key=lambda x: x.score)",
            "async def fetch_all(urls):\n    return await asyncio.gather(*[fetch(u) for u in urls])",
        ],
        "code_output": ["Processing completed in 0.5s", "Successfully fetched 10 items"],
        "method1": ["기본 방법", "동기 처리"],
        "method2": ["최적화 방법", "비동기 처리"],
        "speed1": ["100ms", "500ms"],
        "speed2": ["50ms", "100ms"],
        "memory1": ["50MB", "100MB"],
        "memory2": ["30MB", "50MB"],
        "quote_author": ["Donald Knuth", "Martin Fowler", "Kent Beck"],
        # 문제 해결 템플릿 변수
        "error_name": ["ConnectionError", "MemoryError", "TimeoutError", "ImportError"],
        "error_message": [
            "ConnectionError: Failed to connect to server",
            "MemoryError: Unable to allocate memory",
            "TimeoutError: Operation timed out after 30 seconds",
        ],
        "solution1_command": [
            "pip install --upgrade package-name",
            "export PATH=$PATH:/usr/local/bin",
            "systemctl restart service-name",
        ],
        "solution2_code": [
            "config['timeout'] = 60\nconfig['retries'] = 5",
            "import gc\ngc.collect()",
        ],
        "env1": ["PATH", "PYTHONPATH", "HOME"],
        "env1_val": ["/usr/local/bin", "/app/src", "/home/user"],
        "env2": ["API_KEY", "DATABASE_URL", "LOG_LEVEL"],
        "env2_val": ["your-api-key", "postgresql://localhost/db", "DEBUG"],
        "issue_num": ["123", "456", "789"],
    }

    PAIRED_VARIABLES = {
        "description": [
            ("데이터 처리를 위한 고성능 라이브러리입니다.", "A high-performance library for data processing."),
            ("이미지 분석을 위한 Python 패키지입니다.", "A Python package for image analysis."),
            ("텍스트 파싱 도구 모음입니다.", "A collection of text parsing tools."),
        ],
        "feature1": [
            ("빠른 처리", "Fast Processing"),
            ("간편한 API", "Simple API"),
            ("확장 가능", "Extensible"),
        ],
        "feature1_desc": [
            ("멀티스레딩을 지원합니다", "Supports multi-threading"),
            ("직관적인 인터페이스를 제공합니다", "Provides intuitive interface"),
            ("플러그인 시스템을 지원합니다", "Supports plugin system"),
        ],
        "feature2": [
            ("타입 안전", "Type Safe"),
            ("문서화", "Well Documented"),
            ("테스트 완료", "Fully Tested"),
        ],
        "feature2_desc": [
            ("완전한 타입 힌트를 제공합니다", "Full type hints provided"),
            ("상세한 문서가 제공됩니다", "Comprehensive documentation"),
            ("100% 테스트 커버리지", "100% test coverage"),
        ],
        "func_desc": [
            ("입력 데이터를 처리하고 결과를 반환합니다.", "Processes input data and returns the result."),
            ("이미지를 분석하고 특징을 추출합니다.", "Analyzes an image and extracts features."),
            ("텍스트를 파싱하여 구조화된 데이터로 변환합니다.", "Parses text and converts it to structured data."),
        ],
        "param1_desc": [
            ("처리할 입력 데이터", "Input data to process"),
            ("분석할 데이터", "Data to analyze"),
            ("파싱할 텍스트", "Text to parse"),
            ("파일 경로", "File path"),
        ],
        "param2_desc": [
            ("처리 옵션", "Processing options"),
            ("설정 객체", "Configuration object"),
            ("출력 형식", "Output format"),
            ("인코딩", "Encoding"),
        ],
        "return_desc": [
            ("처리 결과 객체", "Processing result object"),
            ("분석 결과 딕셔너리", "Analysis result dictionary"),
            ("파싱된 문자열 리스트", "List of parsed strings"),
            ("처리된 바이트", "Processed bytes"),
        ],
        "note": [
            ("이 함수는 스레드 안전합니다.", "This function is thread-safe."),
            ("대용량 데이터의 경우 배치 처리를 권장합니다.", "For large data, batch processing is recommended."),
        ],
        # 튜토리얼 템플릿 변수
        "tutorial_title": [
            ("Python 비동기 프로그래밍 시작하기", "Getting Started with Python Async Programming"),
            ("Docker 컨테이너 배포 가이드", "Docker Container Deployment Guide"),
            ("REST API 설계 모범 사례", "REST API Design Best Practices"),
        ],
        "tutorial_intro": [
            ("이 튜토리얼에서는 핵심 개념과 실제 구현 방법을 배웁니다.", "In this tutorial, you will learn core concepts and practical implementation."),
            ("단계별로 따라하면서 실습해보세요.", "Follow along step by step with hands-on practice."),
        ],
        "prereq1": [
            ("Python 3.8 이상", "Python 3.8+"),
            ("Node.js 16 이상", "Node.js 16+"),
            ("Docker Desktop", "Docker Desktop"),
        ],
        "prereq2": [
            ("GitHub", "GitHub"),
            ("Docker Hub", "Docker Hub"),
            ("AWS", "AWS"),
        ],
        "important_note": [
            ("프로덕션 환경에서는 추가 보안 설정이 필요합니다.", "Additional security configuration is required for production."),
            ("테스트 환경에서 먼저 검증하세요.", "Verify in a test environment first."),
        ],
        "timeout_desc": [
            ("요청 타임아웃 (초)", "Request timeout in seconds"),
        ],
        "retries_desc": [
            ("재시도 횟수", "Number of retries"),
        ],
        "debug_desc": [
            ("디버그 모드 활성화", "Enable debug mode"),
        ],
        # 블로그 템플릿 변수
        "blog_title": [
            ("성능 최적화의 비밀", "Secrets of Performance Optimization"),
            ("효율적인 코드 작성법", "Writing Efficient Code"),
            ("개발자가 알아야 할 모범 사례", "Best Practices Every Developer Should Know"),
        ],
        "blog_intro": [
            ("오늘은 실무에서 자주 마주치는 문제와 해결 방법을 공유합니다.", "Today, I'll share common problems encountered in practice and their solutions."),
            ("이 글에서는 제가 프로젝트에서 배운 교훈을 정리했습니다.", "In this post, I've compiled lessons learned from my projects."),
        ],
        "section1_title": [
            ("문제 상황", "The Problem"),
            ("배경", "Background"),
            ("왜 이것이 중요한가?", "Why Does This Matter?"),
        ],
        "section1_content": [
            ("많은 개발자들이 이 문제로 어려움을 겪습니다.", "Many developers struggle with this issue."),
            ("처음에는 간단해 보이지만 복잡한 문제입니다.", "It seems simple at first, but it's a complex problem."),
        ],
        "quote": [
            ("조기 최적화는 모든 악의 근원이다.", "Premature optimization is the root of all evil."),
            ("좋은 코드는 그 자체로 문서가 된다.", "Good code is its own documentation."),
            ("먼저 동작하게 만들고, 그 다음 빠르게 만들어라.", "Make it work, then make it fast."),
        ],
        "point1": [
            ("측정 먼저", "Measure First"),
            ("단순함 유지", "Keep It Simple"),
        ],
        "point1_desc": [
            ("최적화 전에 항상 프로파일링하세요", "Always profile before optimizing"),
            ("복잡한 해결책보다 단순한 해결책을 선호하세요", "Prefer simple solutions over complex ones"),
        ],
        "point2": [
            ("점진적 개선", "Incremental Improvement"),
            ("테스트 작성", "Write Tests"),
        ],
        "point2_desc": [
            ("한 번에 하나씩 개선하세요", "Improve one thing at a time"),
            ("변경 전에 테스트를 작성하세요", "Write tests before making changes"),
        ],
        "point3": [
            ("문서화", "Documentation"),
            ("코드 리뷰", "Code Review"),
        ],
        "point3_desc": [
            ("결정 사항을 기록하세요", "Document your decisions"),
            ("동료의 피드백을 받으세요", "Get feedback from peers"),
        ],
        "conclusion": [
            ("이러한 원칙을 따르면 더 나은 결과를 얻을 수 있습니다.", "Following these principles will lead to better results."),
            ("실천이 중요합니다. 오늘부터 적용해보세요.", "Practice is key. Start applying these today."),
        ],
        # 문제 해결 템플릿 변수
        "symptom": [
            ("애플리케이션이 예기치 않게 종료됩니다.", "The application terminates unexpectedly."),
            ("API 요청이 실패하고 에러가 반환됩니다.", "API requests fail and return errors."),
            ("성능이 급격히 저하됩니다.", "Performance degrades significantly."),
        ],
        "cause1": [
            ("메모리 부족", "Insufficient memory"),
            ("네트워크 연결 문제", "Network connectivity issues"),
            ("잘못된 설정", "Incorrect configuration"),
        ],
        "cause2": [
            ("의존성 충돌", "Dependency conflicts"),
            ("권한 문제", "Permission issues"),
            ("버전 불일치", "Version mismatch"),
        ],
        "cause3": [
            ("리소스 제한", "Resource limits"),
            ("타임아웃 설정", "Timeout settings"),
            ("캐시 문제", "Cache issues"),
        ],
        "cause_note": [
            ("로그를 확인하여 정확한 원인을 파악하세요.", "Check logs to identify the exact cause."),
            ("여러 원인이 복합적으로 작용할 수 있습니다.", "Multiple causes may be involved."),
        ],
        "solution1_title": [
            ("패키지 재설치", "Reinstall Package"),
            ("환경 변수 설정", "Set Environment Variables"),
            ("서비스 재시작", "Restart Service"),
        ],
        "solution2_title": [
            ("설정 파일 수정", "Modify Configuration"),
            ("메모리 정리", "Clear Memory"),
        ],
        "env1_desc": [
            ("실행 파일 경로", "Executable path"),
            ("Python 모듈 경로", "Python module path"),
            ("사용자 홈 디렉토리", "User home directory"),
        ],
        "env2_desc": [
            ("API 인증 키", "API authentication key"),
            ("데이터베이스 연결 문자열", "Database connection string"),
            ("로그 레벨 설정", "Log level setting"),
        ],
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _fill_template(self, template: str, variables: Dict) -> str:
        """템플릿 변수 치환"""
        result = template
        for key, values in variables.items():
            if f"{{{key}}}" in result:
                value = random.choice(values) if isinstance(values, list) else values
                result = result.replace(f"{{{key}}}", str(value))

        # class_name_lower 특수 처리
        if "{class_name_lower}" in result:
            result = result.replace("{class_name_lower}", "obj")

        return result

    def generate_sample(self) -> Tuple[str, str]:
        """샘플 생성"""
        template = random.choice(self.TEMPLATES)

        variables = {}
        for key, values in self.VARIABLES.items():
            variables[key] = random.choice(values)

        for key, pairs in self.PAIRED_VARIABLES.items():
            korean_value, english_value = random.choice(pairs)
            variables[f"{key}_ko"] = korean_value
            variables[f"{key}_en"] = english_value

        korean = self._fill_template(template["korean"], variables)
        english = self._fill_template(template["english"], variables)

        return korean, english

    def generate_dataset(
        self,
        num_samples: int = 50,
        output_path: str = "data/synthetic/template_pairs.jsonl"
    ) -> List[Dict]:
        """데이터셋 생성"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        samples = []

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                korean, english = self.generate_sample()

                sample = {
                    "korean": korean,
                    "english": english,
                    "metadata": {
                        "source": "template",
                        "index": i
                    }
                }
                samples.append(sample)

                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Generated {len(samples)} template samples, saved to {output_path}")
        return samples
