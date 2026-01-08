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

    SYSTEM_PROMPT = """당신은 기술 문서 전문 번역가이자 작가입니다.
한국어와 영어 모두 네이티브 수준으로 구사하며,
마크다운 형식의 기술 문서를 작성하는 데 전문성을 가지고 있습니다.

중요 규칙:
1. 마크다운 문법을 적극적으로 활용하세요 (헤더, 코드 블록, 링크, 리스트, 테이블 등)
2. 코드 예제는 실제로 동작하는 코드를 작성하세요
3. 기술 용어는 정확하게 사용하세요
4. 한국어 버전과 영어 버전의 구조는 동일하게 유지하세요
5. 자연스럽고 읽기 쉬운 문서를 작성하세요"""

    def __init__(
        self,
        provider: Union[str, LLMProvider] = "openai",
        api_key: str = None,
        model: str = None
    ):
        """
        Args:
            provider: LLM 프로바이더 ("openai", "anthropic" 또는 LLMProvider 인스턴스)
            api_key: API 키
            model: 사용할 모델
        """
        if isinstance(provider, LLMProvider):
            self.provider = provider
        elif provider == "openai":
            self.provider = OpenAIProvider(
                api_key=api_key,
                model=model or "gpt-4o"
            )
        elif provider == "anthropic":
            self.provider = AnthropicProvider(
                api_key=api_key,
                model=model or "claude-sonnet-4-20250514"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.model_name = model or (
            "gpt-4o" if provider == "openai" else "claude-sonnet-4-20250514"
        )

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

        prompt = f"""다음 주제와 스타일로 마크다운 기술 문서를 작성해주세요.

주제: {topic}
스타일: {style_desc}

요구사항:
1. 먼저 한국어로 완전한 문서를 작성하세요
2. 그 다음 동일한 내용의 영어 번역을 작성하세요
3. 다음 마크다운 요소를 반드시 포함하세요:
   - 헤더 (# ## ###)
   - 코드 블록 (```)
   - 인라인 코드 (`)
   - 리스트 (- 또는 1.)
   - 볼드/이탤릭 (**bold**, *italic*)
4. 문서 길이: 300-800 단어

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
                    return GeneratedPair(
                        korean=korean,
                        english=english,
                        topic=topic,
                        style=style,
                        model=self.model_name
                    )

        except Exception as e:
            print(f"Generation failed for topic '{topic}': {e}")

        return None

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
    ]

    # 치환 변수들
    VARIABLES = {
        "project_name": ["DataProcessor", "ImageAnalyzer", "TextParser", "APIClient", "FileManager"],
        "package_name": ["dataproc", "imganalyze", "textparse", "apiclient", "filemgr"],
        "class_name": ["Processor", "Analyzer", "Parser", "Client", "Manager"],
        "description_ko": [
            "데이터 처리를 위한 고성능 라이브러리입니다.",
            "이미지 분석을 위한 Python 패키지입니다.",
            "텍스트 파싱 도구 모음입니다.",
        ],
        "description_en": [
            "A high-performance library for data processing.",
            "A Python package for image analysis.",
            "A collection of text parsing tools.",
        ],
        "feature1_ko": ["빠른 처리", "간편한 API", "확장 가능"],
        "feature1_en": ["Fast Processing", "Simple API", "Extensible"],
        "feature1_desc_ko": ["멀티스레딩을 지원합니다", "직관적인 인터페이스를 제공합니다", "플러그인 시스템을 지원합니다"],
        "feature1_desc_en": ["Supports multi-threading", "Provides intuitive interface", "Supports plugin system"],
        "feature2_ko": ["타입 안전", "문서화", "테스트 완료"],
        "feature2_en": ["Type Safe", "Well Documented", "Fully Tested"],
        "feature2_desc_ko": ["완전한 타입 힌트를 제공합니다", "상세한 문서가 제공됩니다", "100% 테스트 커버리지"],
        "feature2_desc_en": ["Full type hints provided", "Comprehensive documentation", "100% test coverage"],
        "func_name": ["process_data", "analyze_image", "parse_text", "fetch_api", "read_file"],
        "func_desc_ko": [
            "입력 데이터를 처리하고 결과를 반환합니다.",
            "이미지를 분석하고 특징을 추출합니다.",
            "텍스트를 파싱하여 구조화된 데이터로 변환합니다.",
        ],
        "func_desc_en": [
            "Processes input data and returns the result.",
            "Analyzes an image and extracts features.",
            "Parses text and converts it to structured data.",
        ],
        "param1": ["input", "data", "text", "path"],
        "type1": ["str", "bytes", "List[str]", "Path"],
        "param1_desc_ko": ["처리할 입력 데이터", "분석할 데이터", "파싱할 텍스트", "파일 경로"],
        "param1_desc_en": ["Input data to process", "Data to analyze", "Text to parse", "File path"],
        "param1_val": ['"hello"', 'b"data"', '["a", "b"]', 'Path("./file")'],
        "param2": ["options", "config", "format", "encoding"],
        "type2": ["dict", "Config", "str", "str"],
        "param2_desc_ko": ["처리 옵션", "설정 객체", "출력 형식", "인코딩"],
        "param2_desc_en": ["Processing options", "Configuration object", "Output format", "Encoding"],
        "param2_val": ['{}', 'Config()', '"json"', '"utf-8"'],
        "return_type": ["Result", "dict", "List[str]", "bytes"],
        "return_desc_ko": ["처리 결과 객체", "분석 결과 딕셔너리", "파싱된 문자열 리스트", "처리된 바이트"],
        "return_desc_en": ["Processing result object", "Analysis result dictionary", "List of parsed strings", "Processed bytes"],
        "note_ko": ["이 함수는 스레드 안전합니다.", "대용량 데이터의 경우 배치 처리를 권장합니다."],
        "note_en": ["This function is thread-safe.", "For large data, batch processing is recommended."],
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

        korean = self._fill_template(template["korean"], self.VARIABLES)
        english = self._fill_template(template["english"], self.VARIABLES)

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
