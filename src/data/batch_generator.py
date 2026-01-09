"""OpenAI Batch API 데이터 생성 모듈

OpenAI Batch API를 사용하여 대량의 데이터를 비용 효율적으로 생성합니다.
- 50% 비용 절감
- 24시간 내 처리 완료 보장
- 최대 50,000 요청/배치 지원

Usage:
    from src.data.batch_generator import BatchDataGenerator

    generator = BatchDataGenerator()

    # 1. 배치 파일 생성 및 제출
    batch_id = generator.submit_batch(num_samples=1000)

    # 2. 상태 확인
    status = generator.check_status(batch_id)

    # 3. 결과 다운로드 (완료 시)
    results = generator.download_results(batch_id, output_path="output.jsonl")
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


@dataclass
class BatchJob:
    """배치 작업 정보"""
    batch_id: str
    input_file_id: str
    status: str
    created_at: datetime
    num_requests: int
    completed: int = 0
    failed: int = 0
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class GeneratedPair:
    """생성된 한영 번역 쌍"""
    korean: str
    english: str
    topic: str
    style: str
    model: str
    request_id: str
    metadata: Dict = field(default_factory=dict)


class BatchDataGenerator:
    """OpenAI Batch API 데이터 생성기

    대량의 데이터를 비용 효율적으로 생성합니다.
    - 50% 비용 절감 (기존 API 대비)
    - 24시간 SLA
    - 별도의 Rate Limit
    """

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
        api_key: str = None,
        model: str = "gpt-4o",
        temperature: float = 0.7
    ):
        """
        Args:
            api_key: OpenAI API 키
            model: 사용할 모델
            temperature: 생성 온도
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self._batch_jobs: Dict[str, BatchJob] = {}

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

    def prepare_batch_file(
        self,
        num_samples: int,
        output_path: str = None,
        topics: List[str] = None,
        styles: List[str] = None,
        seed: int = None
    ) -> Tuple[str, List[Dict]]:
        """Batch 입력 파일 생성

        Args:
            num_samples: 생성할 샘플 수
            output_path: 입력 파일 저장 경로 (None이면 자동 생성)
            topics: 사용할 주제 리스트
            styles: 사용할 스타일 리스트
            seed: 랜덤 시드

        Returns:
            (파일 경로, 요청 메타데이터 리스트)
        """
        if seed is not None:
            random.seed(seed)

        topics = topics or self.TOPICS
        styles = styles or list(self.STYLES.keys())

        # 기본 경로 설정 (현재 파일 기준 상대 경로)
        if output_path is None:
            # 프로젝트 루트 찾기
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent  # src/data -> src -> project_root
            output_path = project_root / "data" / "synthetic" / "batch_input.jsonl"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        requests_metadata = []

        print(f"Preparing batch input file with {num_samples} requests...")

        with open(output_path, 'w', encoding='utf-8') as f:
            iterator = range(num_samples)
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, desc="Creating requests")

            for i in iterator:
                topic = random.choice(topics)
                style = random.choice(styles)
                prompt = self._create_prompt(topic, style)

                # Batch API 요청 형식
                request = {
                    "custom_id": f"request-{i:06d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": self.temperature
                    }
                }

                f.write(json.dumps(request) + '\n')

                # 메타데이터 저장 (나중에 결과 파싱에 사용)
                requests_metadata.append({
                    "custom_id": f"request-{i:06d}",
                    "topic": topic,
                    "style": style
                })

        # 파일 정보 출력
        file_size = output_path.stat().st_size
        print(f"\n=== Batch Input File Created ===")
        print(f"Path: {output_path}")
        print(f"Size: {file_size:,} bytes")
        print(f"Requests: {num_samples}")

        # 첫 번째 요청 샘플 출력 (디버깅용)
        with open(output_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            print(f"\nSample request (first line):")
            print(f"  custom_id: {sample.get('custom_id')}")
            print(f"  method: {sample.get('method')}")
            print(f"  url: {sample.get('url')}")
            print(f"  body.model: {sample.get('body', {}).get('model')}")
            print(f"  body.messages: {len(sample.get('body', {}).get('messages', []))} messages")

        return str(output_path), requests_metadata

    def validate_batch_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """배치 입력 파일 검증

        Args:
            file_path: 검증할 JSONL 파일 경로

        Returns:
            (is_valid, errors) 튜플
        """
        errors = []
        line_count = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line_count = i
                    if not line.strip():
                        continue

                    try:
                        request = json.loads(line)

                        # 필수 필드 확인
                        if "custom_id" not in request:
                            errors.append(f"Line {i}: Missing 'custom_id'")
                        if "method" not in request:
                            errors.append(f"Line {i}: Missing 'method'")
                        if "url" not in request:
                            errors.append(f"Line {i}: Missing 'url'")
                        if "body" not in request:
                            errors.append(f"Line {i}: Missing 'body'")

                        # body 필드 확인
                        if "body" in request:
                            body = request["body"]
                            if "model" not in body:
                                errors.append(f"Line {i}: Missing 'model' in body")
                            if "messages" not in body:
                                errors.append(f"Line {i}: Missing 'messages' in body")

                    except json.JSONDecodeError as e:
                        errors.append(f"Line {i}: Invalid JSON - {e}")

        except FileNotFoundError:
            errors.append(f"File not found: {file_path}")

        is_valid = len(errors) == 0

        print(f"\n=== Batch File Validation ===")
        print(f"File: {file_path}")
        print(f"Total lines: {line_count}")
        print(f"Valid: {'✅ Yes' if is_valid else '❌ No'}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for error in errors[:10]:  # 최대 10개만 표시
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")

        return is_valid, errors

    def submit_batch(
        self,
        num_samples: int = None,
        input_file_path: str = None,
        metadata_path: str = None,
        topics: List[str] = None,
        styles: List[str] = None
    ) -> str:
        """Batch 작업 제출

        Args:
            num_samples: 생성할 샘플 수 (input_file_path가 없을 때)
            input_file_path: 이미 생성된 입력 파일 경로
            metadata_path: 메타데이터 저장 경로
            topics: 사용할 주제 리스트
            styles: 사용할 스타일 리스트

        Returns:
            batch_id
        """
        # 입력 파일 준비
        if input_file_path is None:
            if num_samples is None:
                raise ValueError("Either num_samples or input_file_path must be provided")

            input_file_path, requests_metadata = self.prepare_batch_file(
                num_samples=num_samples,
                topics=topics,
                styles=styles
            )

            # 메타데이터 저장
            if metadata_path is None:
                metadata_path = input_file_path.replace('.jsonl', '_metadata.json')

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(requests_metadata, f, ensure_ascii=False, indent=2)
            print(f"Metadata saved: {metadata_path}")
        else:
            requests_metadata = []

        # 입력 파일 검증
        print("\nValidating batch input file...")
        is_valid, validation_errors = self.validate_batch_file(input_file_path)
        if not is_valid:
            raise ValueError(f"Batch input file validation failed. Fix errors and try again.")

        # 파일 업로드
        print("\nUploading batch file to OpenAI...")
        with open(input_file_path, 'rb') as f:
            batch_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"File uploaded: {batch_file.id}")

        # Batch 작업 생성
        print("Creating batch job...")
        batch = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Synthetic data generation - {num_samples or 'custom'} samples"
            }
        )

        # 작업 정보 저장
        job = BatchJob(
            batch_id=batch.id,
            input_file_id=batch_file.id,
            status=batch.status,
            created_at=datetime.now(),
            num_requests=num_samples or len(requests_metadata),
            metadata={"input_file_path": input_file_path, "metadata_path": metadata_path}
        )
        self._batch_jobs[batch.id] = job

        print(f"\n=== Batch Job Submitted ===")
        print(f"Batch ID: {batch.id}")
        print(f"Status: {batch.status}")
        print(f"Input file: {batch_file.id}")
        print(f"\nTo check status, run:")
        print(f"  generator.check_status('{batch.id}')")

        return batch.id

    def check_status(self, batch_id: str, verbose: bool = True) -> Dict:
        """Batch 상태 확인

        Args:
            batch_id: 배치 작업 ID
            verbose: 상세 정보 출력 여부

        Returns:
            상태 정보 딕셔너리
        """
        batch = self.client.batches.retrieve(batch_id)

        status = {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed": batch.request_counts.completed if batch.request_counts else 0,
            "failed": batch.request_counts.failed if batch.request_counts else 0,
            "total": batch.request_counts.total if batch.request_counts else 0,
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "input_file_id": batch.input_file_id,
            "errors": None,
        }

        # 에러 정보 추출
        if hasattr(batch, 'errors') and batch.errors:
            status["errors"] = batch.errors

        # 로컬 캐시 업데이트
        if batch_id in self._batch_jobs:
            self._batch_jobs[batch_id].status = batch.status
            self._batch_jobs[batch_id].completed = status["completed"]
            self._batch_jobs[batch_id].failed = status["failed"]
            self._batch_jobs[batch_id].output_file_id = batch.output_file_id
            self._batch_jobs[batch_id].error_file_id = batch.error_file_id

        # 진행률 계산
        if status["total"] > 0:
            progress = (status["completed"] + status["failed"]) / status["total"] * 100
            status["progress"] = f"{progress:.1f}%"

        if verbose:
            print(f"\n=== Batch Status ===")
            print(f"Batch ID: {batch_id}")
            print(f"Status: {status['status']}")
            print(f"Input File: {status['input_file_id']}")
            print(f"Progress: {status.get('progress', 'N/A')}")
            print(f"Completed: {status['completed']}/{status['total']}")
            print(f"Failed: {status['failed']}")

            # 에러 정보 출력
            if status["errors"]:
                print(f"\n⚠️  Errors:")
                if hasattr(status["errors"], 'data'):
                    for error in status["errors"].data:
                        print(f"  - {error.code}: {error.message}")
                else:
                    print(f"  {status['errors']}")

            # 상태별 안내 메시지
            if status["status"] == "validating":
                print(f"\n📋 Batch is being validated. This may take a few minutes...")
            elif status["status"] == "in_progress":
                print(f"\n🚀 Batch is processing...")
            elif status["status"] == "completed":
                print(f"\n✅ Batch completed! Use download_results() to get results.")
            elif status["status"] == "failed":
                print(f"\n❌ Batch failed. Check errors above.")
            elif status["status"] == "expired":
                print(f"\n⏰ Batch expired (24h limit reached).")
            elif status["status"] == "cancelled":
                print(f"\n🚫 Batch was cancelled.")

        return status

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 30,
        timeout: int = 86400  # 24 hours
    ) -> Dict:
        """Batch 완료 대기

        Args:
            batch_id: 배치 작업 ID
            poll_interval: 상태 확인 간격 (초)
            timeout: 최대 대기 시간 (초)

        Returns:
            최종 상태 정보
        """
        print(f"\nWaiting for batch {batch_id} to complete...")
        print(f"Polling interval: {poll_interval}s")
        print(f"Timeout: {timeout}s")
        print()

        start_time = time.time()
        last_completed = 0
        last_status = None

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Batch job timed out after {timeout}s")

            # 상태 변경 시에만 상세 출력
            status = self.check_status(batch_id, verbose=False)

            # 상태 변경 시 출력
            if status["status"] != last_status:
                elapsed_min = elapsed / 60
                print(f"[{elapsed_min:.1f}m] Status: {status['status']} | "
                      f"Progress: {status['completed']}/{status['total']}")
                last_status = status["status"]

                # 에러가 있으면 출력
                if status.get("errors"):
                    print(f"  ⚠️  Errors detected:")
                    if hasattr(status["errors"], 'data'):
                        for error in status["errors"].data:
                            print(f"    - {error.code}: {error.message}")

            # 진행 상황 출력 (처리 중인 경우)
            elif status["completed"] > last_completed:
                rate = (status["completed"] - last_completed) / poll_interval
                elapsed_min = elapsed / 60
                print(f"[{elapsed_min:.1f}m] Progress: {status['completed']}/{status['total']} "
                      f"(~{rate:.1f} req/s)")
                last_completed = status["completed"]

            if status["status"] == "completed":
                print(f"\n✅ Batch completed successfully!")
                print(f"   Total time: {elapsed/60:.1f} minutes")
                return status
            elif status["status"] == "failed":
                # 실패 시 상세 정보 출력
                self.check_status(batch_id, verbose=True)
                raise RuntimeError(f"Batch job failed. Check errors above.")
            elif status["status"] == "cancelled":
                raise RuntimeError(f"Batch job was cancelled")
            elif status["status"] == "expired":
                raise RuntimeError(f"Batch job expired (24h limit reached)")

            # 대기
            time.sleep(poll_interval)

    def _parse_response(self, response_text: str, topic: str, style: str, request_id: str) -> Optional[GeneratedPair]:
        """응답 파싱"""
        if "---KOREAN---" in response_text and "---ENGLISH---" in response_text:
            korean_start = response_text.find("---KOREAN---") + len("---KOREAN---")
            korean_end = response_text.find("---ENGLISH---")
            english_start = korean_end + len("---ENGLISH---")
            english_end = response_text.find("---END---") if "---END---" in response_text else len(response_text)

            korean = response_text[korean_start:korean_end].strip()
            english = response_text[english_start:english_end].strip()

            if korean and english:
                return GeneratedPair(
                    korean=korean,
                    english=english,
                    topic=topic,
                    style=style,
                    model=self.model,
                    request_id=request_id
                )
        return None

    def download_results(
        self,
        batch_id: str,
        output_path: str = "data/synthetic/batch_output.jsonl",
        metadata_path: str = None
    ) -> List[GeneratedPair]:
        """결과 다운로드 및 파싱

        Args:
            batch_id: 배치 작업 ID
            output_path: 결과 저장 경로
            metadata_path: 메타데이터 파일 경로 (topic, style 정보)

        Returns:
            생성된 쌍 리스트
        """
        # 상태 확인
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch not completed. Current status: {batch.status}")

        if not batch.output_file_id:
            raise ValueError("No output file available")

        # 메타데이터 로드
        metadata_map = {}
        if metadata_path:
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                    for item in metadata_list:
                        metadata_map[item["custom_id"]] = item
            except FileNotFoundError:
                print(f"Warning: Metadata file not found: {metadata_path}")

        # 로컬 캐시에서 메타데이터 경로 확인
        if not metadata_map and batch_id in self._batch_jobs:
            cached_metadata_path = self._batch_jobs[batch_id].metadata.get("metadata_path")
            if cached_metadata_path:
                try:
                    with open(cached_metadata_path, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                        for item in metadata_list:
                            metadata_map[item["custom_id"]] = item
                except FileNotFoundError:
                    pass

        # 결과 파일 다운로드
        print(f"Downloading results from {batch.output_file_id}...")
        result_content = self.client.files.content(batch.output_file_id)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        successful = 0
        failed = 0

        with open(output_path, 'w', encoding='utf-8') as f:
            for line in result_content.text.strip().split('\n'):
                if not line:
                    continue

                try:
                    result = json.loads(line)
                    custom_id = result.get("custom_id", "")

                    # 메타데이터에서 topic, style 가져오기
                    meta = metadata_map.get(custom_id, {})
                    topic = meta.get("topic", "unknown")
                    style = meta.get("style", "unknown")

                    # 응답 추출
                    if result.get("response", {}).get("status_code") == 200:
                        body = result["response"]["body"]
                        content = body["choices"][0]["message"]["content"]

                        pair = self._parse_response(content, topic, style, custom_id)

                        if pair:
                            results.append(pair)
                            successful += 1

                            # JSONL 형식으로 저장
                            data = {
                                "korean": pair.korean,
                                "english": pair.english,
                                "metadata": {
                                    "topic": pair.topic,
                                    "style": pair.style,
                                    "model": pair.model,
                                    "source": "synthetic_batch",
                                    "request_id": pair.request_id
                                }
                            }
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
                        else:
                            failed += 1
                    else:
                        failed += 1
                        error = result.get("error", {})
                        print(f"Request {custom_id} failed: {error}")

                except json.JSONDecodeError as e:
                    print(f"Failed to parse result line: {e}")
                    failed += 1

        # 에러 파일 처리
        if batch.error_file_id:
            print(f"\nDownloading error file...")
            error_content = self.client.files.content(batch.error_file_id)
            error_path = str(output_path).replace('.jsonl', '_errors.jsonl')
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(error_content.text)
            print(f"Errors saved to: {error_path}")

        print(f"\n=== Download Complete ===")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output: {output_path}")

        return results

    def list_batches(self, limit: int = 10) -> List[Dict]:
        """최근 배치 작업 목록 조회

        Args:
            limit: 조회할 최대 개수

        Returns:
            배치 작업 목록
        """
        batches = self.client.batches.list(limit=limit)

        results = []
        print(f"\n=== Recent Batch Jobs ===")
        for batch in batches.data:
            info = {
                "batch_id": batch.id,
                "status": batch.status,
                "created_at": batch.created_at,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "total": batch.request_counts.total if batch.request_counts else 0,
            }
            results.append(info)
            print(f"  {batch.id}: {batch.status} ({info['completed']}/{info['total']})")

        return results

    def cancel_batch(self, batch_id: str) -> Dict:
        """배치 작업 취소

        Args:
            batch_id: 배치 작업 ID

        Returns:
            취소된 배치 정보
        """
        batch = self.client.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancellation requested. Status: {batch.status}")
        return {"batch_id": batch.id, "status": batch.status}


def run_batch_generation(
    num_samples: int,
    output_path: str = None,
    api_key: str = None,
    model: str = "gpt-4o",
    wait: bool = True,
    poll_interval: int = 30
) -> Tuple[str, Optional[List[GeneratedPair]]]:
    """배치 생성 실행 (CLI용 래퍼)

    Args:
        num_samples: 생성할 샘플 수
        output_path: 결과 저장 경로
        api_key: OpenAI API 키
        model: 사용할 모델
        wait: 완료까지 대기할지 여부
        poll_interval: 상태 확인 간격

    Returns:
        (batch_id, results) - wait=False면 results는 None
    """
    output_path = output_path or f"data/synthetic/batch_output.jsonl"

    generator = BatchDataGenerator(api_key=api_key, model=model)

    # 배치 제출
    batch_id = generator.submit_batch(num_samples=num_samples)

    if not wait:
        print(f"\nBatch submitted. To check status later:")
        print(f"  python -c \"from src.data.batch_generator import BatchDataGenerator; g = BatchDataGenerator(); g.check_status('{batch_id}')\"")
        return batch_id, None

    # 완료 대기
    generator.wait_for_completion(batch_id, poll_interval=poll_interval)

    # 결과 다운로드
    results = generator.download_results(batch_id, output_path=output_path)

    return batch_id, results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch_generator.py submit <num_samples>")
        print("  python batch_generator.py status <batch_id>")
        print("  python batch_generator.py download <batch_id> [output_path]")
        print("  python batch_generator.py list")
        sys.exit(1)

    command = sys.argv[1]
    generator = BatchDataGenerator()

    if command == "submit":
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        generator.submit_batch(num_samples=num_samples)

    elif command == "status":
        batch_id = sys.argv[2]
        generator.check_status(batch_id)

    elif command == "download":
        batch_id = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        generator.download_results(batch_id, output_path=output_path)

    elif command == "list":
        generator.list_batches()

    elif command == "cancel":
        batch_id = sys.argv[2]
        generator.cancel_batch(batch_id)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
