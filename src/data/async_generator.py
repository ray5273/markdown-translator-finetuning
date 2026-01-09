"""비동기 병렬 데이터 생성 모듈

asyncio를 사용하여 OpenAI/Anthropic API를 병렬로 호출하여 데이터 생성 속도를 향상시킵니다.

Usage:
    from src.data.async_generator import AsyncSyntheticDataGenerator

    generator = AsyncSyntheticDataGenerator(provider="openai", max_concurrent=50)
    results = await generator.generate_dataset(num_samples=1000)
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    from tqdm.asyncio import tqdm_asyncio
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_ASYNC_AVAILABLE = True
except ImportError:
    OPENAI_ASYNC_AVAILABLE = False

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_ASYNC_AVAILABLE = True
except ImportError:
    ANTHROPIC_ASYNC_AVAILABLE = False

# Local LLM providers
LOCAL_LLM_ASYNC_AVAILABLE = False
try:
    from .local_llm_provider import (
        AsyncOllamaProvider,
        AsyncVLLMProvider,
        check_ollama_available,
        check_vllm_available,
    )
    LOCAL_LLM_ASYNC_AVAILABLE = True
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
            AsyncOllamaProvider = _local_llm_module.AsyncOllamaProvider
            AsyncVLLMProvider = _local_llm_module.AsyncVLLMProvider
            check_ollama_available = _local_llm_module.check_ollama_available
            check_vllm_available = _local_llm_module.check_vllm_available
            LOCAL_LLM_ASYNC_AVAILABLE = True
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
    metadata: Dict = field(default_factory=dict)


@dataclass
class GenerationStats:
    """생성 통계"""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    start_time: float = 0
    end_time: float = 0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.successful / self.total_requests * 100

    @property
    def requests_per_second(self) -> float:
        if self.duration == 0:
            return 0
        return self.successful / self.duration


class AsyncLLMProvider(ABC):
    """비동기 LLM 프로바이더 추상 클래스"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성 (비동기)"""
        pass

    @abstractmethod
    async def close(self):
        """리소스 정리"""
        pass


class AsyncOpenAIProvider(AsyncLLMProvider):
    """비동기 OpenAI API 프로바이더"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        if not OPENAI_ASYNC_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        raise last_error

    async def close(self):
        await self.client.close()


class AsyncAnthropicProvider(AsyncLLMProvider):
    """비동기 Anthropic API 프로바이더"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        if not ANTHROPIC_ASYNC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return response.content[0].text
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        raise last_error

    async def close(self):
        await self.client.close()


class AsyncSyntheticDataGenerator:
    """비동기 합성 데이터 생성기

    asyncio를 사용하여 여러 API 요청을 병렬로 처리합니다.
    Semaphore를 통해 동시 요청 수를 제어합니다.
    """

    # 기술 문서 주제들
    TOPICS = [
        "Python 비동기 프로그래밍", "JavaScript 프레임워크 비교", "Rust 메모리 관리",
        "Go 동시성 패턴", "TypeScript 타입 시스템", "React 상태 관리",
        "Vue.js 컴포넌트 설계", "Node.js 성능 최적화", "Docker 컨테이너화",
        "Kubernetes 클러스터 관리", "딥러닝 모델 학습", "자연어 처리 기초",
        "트랜스포머 아키텍처", "모델 파인튜닝 가이드", "MLOps 파이프라인",
        "LLM 프롬프트 엔지니어링", "PostgreSQL 최적화", "MongoDB 스키마 설계",
        "Redis 캐싱 전략", "SQL 쿼리 튜닝", "CI/CD 파이프라인 구축",
        "AWS 서버리스 아키텍처", "Terraform 인프라 코드", "Nginx 설정 가이드",
        "Git 브랜치 전략", "API 설계 원칙", "마이크로서비스 아키텍처",
        "보안 모범 사례", "성능 모니터링", "로깅 시스템 구축",
    ]

    STYLES = {
        "readme": "프로젝트 README 문서 스타일 (설치 방법, 사용법, 예제 포함)",
        "tutorial": "단계별 튜토리얼 스타일 (상세한 설명과 코드 예제)",
        "api_doc": "API 문서 스타일 (함수 시그니처, 파라미터, 반환값 설명)",
        "blog": "기술 블로그 스타일 (친근한 어조, 경험 공유)",
        "reference": "레퍼런스 문서 스타일 (간결하고 정확한 설명)",
        "troubleshooting": "문제 해결 가이드 스타일 (증상, 원인, 해결책)",
    }

    STYLE_REQUIRED_ELEMENTS = {
        "readme": ["headers", "code_blocks", "lists", "links", "bold"],
        "tutorial": ["headers", "code_blocks", "inline_codes", "lists", "links", "blockquotes"],
        "api_doc": ["headers", "code_blocks", "inline_codes", "tables", "lists"],
        "blog": ["headers", "code_blocks", "links", "bold", "italic", "blockquotes"],
        "reference": ["headers", "tables", "inline_codes", "lists", "links"],
        "troubleshooting": ["headers", "code_blocks", "lists", "blockquotes", "bold"],
    }

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
    }

    SYSTEM_PROMPT = """당신은 기술 문서 전문 번역가이자 작가입니다.
한국어와 영어 모두 네이티브 수준으로 구사하며,
마크다운 형식의 기술 문서를 작성하는 데 전문성을 가지고 있습니다.

중요 규칙:
1. 마크다운 문법을 다양하고 적극적으로 활용하세요
2. 코드 예제는 실제로 동작하는 코드를 작성하세요
3. 기술 용어는 정확하게 사용하세요
4. 한국어 버전과 영어 버전의 구조와 마크다운 요소는 동일하게 유지하세요
5. 자연스럽고 읽기 쉬운 문서를 작성하세요"""

    def __init__(
        self,
        provider: str = "openai",
        api_key: str = None,
        model: str = None,
        max_concurrent: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        base_url: str = None,
        **kwargs
    ):
        """
        Args:
            provider: LLM 프로바이더 ("openai", "anthropic", "ollama", "vllm")
            api_key: API 키 (OpenAI, Anthropic용)
            model: 사용할 모델
            max_concurrent: 최대 동시 요청 수
            max_retries: 실패 시 재시도 횟수
            retry_delay: 재시도 기본 대기 시간 (초)
            base_url: 서버 URL (Ollama, vLLM용)
            **kwargs: 추가 설정 (timeout, max_tokens 등)
        """
        self.provider_name = provider
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = base_url
        self.extra_kwargs = kwargs

        if provider == "openai":
            self.model_name = model or "gpt-4o"
            self._provider_class = AsyncOpenAIProvider
        elif provider == "anthropic":
            self.model_name = model or "claude-sonnet-4-20250514"
            self._provider_class = AsyncAnthropicProvider
        elif provider == "ollama":
            if not LOCAL_LLM_ASYNC_AVAILABLE:
                raise ImportError(
                    "Local LLM provider not available. "
                    "Make sure httpx package is installed: pip install httpx"
                )
            self.model_name = model or "llama3.1:8b"
            self._provider_class = AsyncOllamaProvider
        elif provider == "vllm":
            if not LOCAL_LLM_ASYNC_AVAILABLE:
                raise ImportError(
                    "Local LLM provider not available. "
                    "Make sure openai package is installed: pip install openai"
                )
            self.model_name = model  # vLLM은 서버에서 모델 감지 가능
            self._provider_class = AsyncVLLMProvider
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, ollama, vllm")

        self._provider: Optional[AsyncLLMProvider] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.stats = GenerationStats()

    async def _get_provider(self) -> AsyncLLMProvider:
        """프로바이더 인스턴스 반환 (lazy initialization)"""
        if self._provider is None:
            if self.provider_name in ("openai", "anthropic"):
                # API 기반 프로바이더
                self._provider = self._provider_class(
                    api_key=self.api_key,
                    model=self.model_name,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
            elif self.provider_name == "ollama":
                # Ollama 프로바이더
                self._provider = self._provider_class(
                    base_url=self.base_url or "http://localhost:11434",
                    model=self.model_name,
                    temperature=self.extra_kwargs.get("temperature", 0.7),
                    max_tokens=self.extra_kwargs.get("max_tokens", 4096),
                    timeout=self.extra_kwargs.get("timeout", 120.0),
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
            elif self.provider_name == "vllm":
                # vLLM 프로바이더
                self._provider = self._provider_class(
                    base_url=self.base_url or "http://localhost:8000",
                    model=self.model_name,
                    temperature=self.extra_kwargs.get("temperature", 0.7),
                    max_tokens=self.extra_kwargs.get("max_tokens", 4096),
                    api_key=self.extra_kwargs.get("api_key", "EMPTY"),
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay
                )
        return self._provider

    def _get_required_elements_prompt(self, style: str) -> str:
        """스타일별 필수 마크다운 요소 프롬프트 생성"""
        required = self.STYLE_REQUIRED_ELEMENTS.get(style, self.STYLE_REQUIRED_ELEMENTS["tutorial"])
        elements_desc = []
        for elem in required:
            if elem in self.MARKDOWN_ELEMENT_DESCRIPTIONS:
                elements_desc.append(f"   - {self.MARKDOWN_ELEMENT_DESCRIPTIONS[elem]}")
        return "\n".join(elements_desc)

    def _create_prompt(self, topic: str, style: str) -> str:
        """생성 프롬프트 생성"""
        style_desc = self.STYLES.get(style, self.STYLES["tutorial"])
        required_elements = self._get_required_elements_prompt(style)

        return f"""다음 주제와 스타일로 마크다운 기술 문서를 작성해주세요.

주제: {topic}
스타일: {style_desc}

요구사항:
1. 먼저 한국어로 완전한 문서를 작성하세요
2. 그 다음 동일한 내용의 영어 번역을 작성하세요
3. 다음 마크다운 요소를 반드시 포함하세요:
{required_elements}
4. 문서 길이: 400-1000 단어
5. 한국어와 영어 버전은 동일한 마크다운 구조를 유지하세요

출력 형식:
---KOREAN---
[한국어 마크다운 문서]
---ENGLISH---
[영어 마크다운 문서]
---END---"""

    def _parse_response(self, response: str, topic: str, style: str) -> Optional[GeneratedPair]:
        """응답 파싱"""
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
        return None

    async def generate_single(
        self,
        topic: str,
        style: str,
        request_id: int = 0
    ) -> Tuple[int, Optional[GeneratedPair]]:
        """단일 샘플 생성 (동시성 제어 포함)"""
        async with self._semaphore:
            provider = await self._get_provider()
            prompt = self._create_prompt(topic, style)

            try:
                response = await provider.generate(prompt, self.SYSTEM_PROMPT)
                pair = self._parse_response(response, topic, style)
                if pair:
                    self.stats.successful += 1
                    return (request_id, pair)
                else:
                    self.stats.failed += 1
                    return (request_id, None)
            except Exception as e:
                self.stats.failed += 1
                print(f"Request {request_id} failed: {e}")
                return (request_id, None)

    async def generate_dataset(
        self,
        num_samples: int = 100,
        output_path: str = "data/synthetic/async_generated.jsonl",
        topics: List[str] = None,
        styles: List[str] = None,
        save_interval: int = 10
    ) -> List[GeneratedPair]:
        """데이터셋 병렬 생성

        Args:
            num_samples: 생성할 샘플 수
            output_path: 저장 경로
            topics: 사용할 주제 리스트
            styles: 사용할 스타일 리스트
            save_interval: 중간 저장 간격

        Returns:
            생성된 쌍 리스트
        """
        topics = topics or self.TOPICS
        styles = styles or list(self.STYLES.keys())

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 초기화
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self.stats = GenerationStats(total_requests=num_samples)
        self.stats.start_time = time.time()

        # 태스크 생성
        tasks = []
        for i in range(num_samples):
            topic = random.choice(topics)
            style = random.choice(styles)
            tasks.append(self.generate_single(topic, style, i))

        # 병렬 실행 with progress bar
        results: List[GeneratedPair] = []

        print(f"\n=== Async Parallel Generation ===")
        print(f"Total samples: {num_samples}")
        print(f"Max concurrent: {self.max_concurrent}")
        print(f"Provider: {self.provider_name} ({self.model_name})")
        print()

        # 결과를 파일에 스트리밍 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            if TQDM_AVAILABLE:
                # tqdm을 사용한 진행률 표시
                completed = 0
                for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Generating"):
                    request_id, pair = await coro
                    completed += 1

                    if pair:
                        results.append(pair)
                        data = {
                            "korean": pair.korean,
                            "english": pair.english,
                            "metadata": {
                                "topic": pair.topic,
                                "style": pair.style,
                                "model": pair.model,
                                "source": "synthetic_async",
                                "request_id": request_id
                            }
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')

                        # 주기적으로 flush
                        if completed % save_interval == 0:
                            f.flush()
            else:
                # tqdm 없이 진행
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    request_id, pair = await coro

                    if pair:
                        results.append(pair)
                        data = {
                            "korean": pair.korean,
                            "english": pair.english,
                            "metadata": {
                                "topic": pair.topic,
                                "style": pair.style,
                                "model": pair.model,
                                "source": "synthetic_async",
                                "request_id": request_id
                            }
                        }
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')

                    if (i + 1) % 10 == 0:
                        print(f"Progress: {i + 1}/{num_samples}")
                        f.flush()

        self.stats.end_time = time.time()

        # 통계 출력
        print(f"\n=== Generation Complete ===")
        print(f"Successful: {self.stats.successful}/{self.stats.total_requests}")
        print(f"Failed: {self.stats.failed}")
        print(f"Success rate: {self.stats.success_rate:.1f}%")
        print(f"Duration: {self.stats.duration:.1f}s")
        print(f"Speed: {self.stats.requests_per_second:.2f} req/s")
        print(f"Output: {output_path}")

        return results

    async def close(self):
        """리소스 정리"""
        if self._provider:
            await self._provider.close()
            self._provider = None


async def main_async(
    provider: str = "openai",
    num_samples: int = 100,
    output_path: str = None,
    api_key: str = None,
    model: str = None,
    max_concurrent: int = 50,
    base_url: str = None,
    **kwargs
):
    """비동기 메인 함수"""
    output_path = output_path or f"data/synthetic/{provider}_async_pairs.jsonl"

    generator = AsyncSyntheticDataGenerator(
        provider=provider,
        api_key=api_key,
        model=model,
        max_concurrent=max_concurrent,
        base_url=base_url,
        **kwargs
    )

    try:
        results = await generator.generate_dataset(
            num_samples=num_samples,
            output_path=output_path
        )
        return results
    finally:
        await generator.close()


def run_async_generation(
    provider: str = "openai",
    num_samples: int = 100,
    output_path: str = None,
    api_key: str = None,
    model: str = None,
    max_concurrent: int = 50,
    base_url: str = None,
    **kwargs
):
    """동기 래퍼 함수 (CLI에서 사용)"""
    return asyncio.run(main_async(
        provider=provider,
        num_samples=num_samples,
        output_path=output_path,
        api_key=api_key,
        model=model,
        max_concurrent=max_concurrent,
        base_url=base_url,
        **kwargs
    ))


if __name__ == "__main__":
    # 테스트 실행
    import sys

    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    run_async_generation(provider=provider, num_samples=num_samples)
