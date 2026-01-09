"""Local LLM 프로바이더 모듈

Ollama와 vLLM을 사용하여 로컬에서 LLM을 실행하고 합성 데이터를 생성합니다.

지원하는 프로바이더:
- Ollama: 가장 쉬운 설치, REST API 기반
- vLLM: 고성능, OpenAI 호환 API

Usage:
    # Ollama
    provider = OllamaProvider(model="llama3.1:8b")
    response = provider.generate("Hello, world!")

    # vLLM (OpenAI 호환)
    provider = VLLMProvider(base_url="http://localhost:8000", model="meta-llama/Llama-3.1-8B-Instruct")
    response = provider.generate("Hello, world!")

    # 비동기 버전
    provider = AsyncOllamaProvider(model="llama3.1:8b")
    response = await provider.generate("Hello, world!")
"""

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# HTTP 클라이언트
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# OpenAI 클라이언트 (vLLM용)
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class LocalLLMConfig:
    """Local LLM 설정"""
    provider: str = "ollama"  # ollama, vllm
    base_url: str = None  # 서버 URL
    model: str = None  # 모델 이름
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0  # 타임아웃 (초)

    def __post_init__(self):
        # 기본값 설정
        if self.base_url is None:
            if self.provider == "ollama":
                self.base_url = "http://localhost:11434"
            elif self.provider == "vllm":
                self.base_url = "http://localhost:8000"

        if self.model is None:
            if self.provider == "ollama":
                self.model = "llama3.1:8b"
            elif self.provider == "vllm":
                self.model = "meta-llama/Llama-3.1-8B-Instruct"


class LLMProvider(ABC):
    """LLM 프로바이더 추상 클래스 (동기)"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성"""
        pass


class AsyncLLMProvider(ABC):
    """LLM 프로바이더 추상 클래스 (비동기)"""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성 (비동기)"""
        pass

    @abstractmethod
    async def close(self):
        """리소스 정리"""
        pass


# =============================================================================
# Ollama Provider
# =============================================================================

class OllamaProvider(LLMProvider):
    """Ollama API 프로바이더 (동기)

    Ollama REST API를 사용하여 로컬 LLM과 통신합니다.

    설치:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.1:8b

    Args:
        base_url: Ollama 서버 URL (기본: http://localhost:11434)
        model: 모델 이름 (기본: llama3.1:8b)
        temperature: 샘플링 온도
        max_tokens: 최대 생성 토큰 수
        timeout: 요청 타임아웃 (초)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package not installed. Run: pip install requests")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # 연결 테스트
        self._check_connection()

    def _check_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Ollama 서버에 연결할 수 없습니다: {self.base_url}\n"
                "Ollama가 실행 중인지 확인하세요: ollama serve"
            )
        except Exception as e:
            print(f"Warning: Ollama 연결 확인 실패: {e}")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성"""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result["message"]["content"]

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]


class AsyncOllamaProvider(AsyncLLMProvider):
    """Ollama API 프로바이더 (비동기)

    httpx를 사용한 비동기 Ollama 클라이언트입니다.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package not installed. Run: pip install httpx")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 반환 (lazy initialization)"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성 (비동기)"""
        import asyncio

        client = await self._get_client()
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        raise last_error

    async def close(self):
        """리소스 정리"""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# vLLM Provider (OpenAI 호환 API)
# =============================================================================

class VLLMProvider(LLMProvider):
    """vLLM OpenAI 호환 API 프로바이더 (동기)

    vLLM 서버의 OpenAI 호환 API를 사용합니다.

    vLLM 서버 시작:
        python -m vllm.entrypoints.openai.api_server \\
            --model meta-llama/Llama-3.1-8B-Instruct \\
            --port 8000

    Args:
        base_url: vLLM 서버 URL (기본: http://localhost:8000)
        model: 모델 이름
        temperature: 샘플링 온도
        max_tokens: 최대 생성 토큰 수
        api_key: API 키 (vLLM은 보통 불필요, 기본: "EMPTY")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str = "EMPTY"
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # OpenAI 클라이언트를 vLLM 서버에 연결
        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1"
        )

        # 모델 이름 자동 감지
        if self.model is None:
            self.model = self._detect_model()

    def _detect_model(self) -> str:
        """서버에서 사용 가능한 모델 감지"""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
        except Exception as e:
            print(f"Warning: 모델 자동 감지 실패: {e}")
        return "default"

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        models = self.client.models.list()
        return [m.id for m in models.data]


class AsyncVLLMProvider(AsyncLLMProvider):
    """vLLM OpenAI 호환 API 프로바이더 (비동기)

    AsyncOpenAI 클라이언트를 사용한 비동기 vLLM 연결입니다.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: str = "EMPTY",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # AsyncOpenAI 클라이언트를 vLLM 서버에 연결
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{self.base_url}/v1"
        )

        # 모델 이름 (동기적으로 감지하거나 나중에 설정)
        self._model_detected = False

    async def _ensure_model(self):
        """모델 이름 확인 (필요시 감지)"""
        if self.model is None and not self._model_detected:
            try:
                models = await self.client.models.list()
                if models.data:
                    self.model = models.data[0].id
            except Exception:
                self.model = "default"
            self._model_detected = True

    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """텍스트 생성 (비동기)"""
        import asyncio

        await self._ensure_model()

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
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)

        raise last_error

    async def close(self):
        """리소스 정리"""
        await self.client.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_local_provider(
    provider: str = "ollama",
    base_url: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> LLMProvider:
    """Local LLM 프로바이더 생성 (동기)

    Args:
        provider: 프로바이더 종류 ("ollama" 또는 "vllm")
        base_url: 서버 URL
        model: 모델 이름
        temperature: 샘플링 온도
        max_tokens: 최대 생성 토큰 수
        **kwargs: 추가 설정

    Returns:
        LLMProvider 인스턴스
    """
    config = LocalLLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if provider == "ollama":
        return OllamaProvider(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=kwargs.get("timeout", 120.0)
        )
    elif provider == "vllm":
        return VLLMProvider(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=kwargs.get("api_key", "EMPTY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, vllm")


def create_async_local_provider(
    provider: str = "ollama",
    base_url: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> AsyncLLMProvider:
    """Local LLM 프로바이더 생성 (비동기)

    Args:
        provider: 프로바이더 종류 ("ollama" 또는 "vllm")
        base_url: 서버 URL
        model: 모델 이름
        temperature: 샘플링 온도
        max_tokens: 최대 생성 토큰 수
        **kwargs: 추가 설정

    Returns:
        AsyncLLMProvider 인스턴스
    """
    config = LocalLLMConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if provider == "ollama":
        return AsyncOllamaProvider(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=kwargs.get("timeout", 120.0),
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 1.0)
        )
    elif provider == "vllm":
        return AsyncVLLMProvider(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=kwargs.get("api_key", "EMPTY"),
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 1.0)
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: ollama, vllm")


# =============================================================================
# Utility Functions
# =============================================================================

def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Ollama 서버 사용 가능 여부 확인"""
    if not REQUESTS_AVAILABLE:
        return False
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_vllm_available(base_url: str = "http://localhost:8000") -> bool:
    """vLLM 서버 사용 가능 여부 확인"""
    if not REQUESTS_AVAILABLE:
        return False
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Ollama에서 사용 가능한 모델 목록"""
    if not REQUESTS_AVAILABLE:
        return []
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def list_vllm_models(base_url: str = "http://localhost:8000") -> List[str]:
    """vLLM에서 사용 가능한 모델 목록"""
    if not OPENAI_AVAILABLE:
        return []
    try:
        client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")
        models = client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return []
