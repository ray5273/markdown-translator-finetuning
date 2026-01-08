"""데이터 수집 모듈

다양한 소스에서 한영 병렬 데이터를 수집합니다.
- GitHub 한국어 기술 문서
- 공개 병렬 코퍼스 (AI Hub, OPUS 등)
- 웹 크롤링
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
import requests
from tqdm import tqdm


@dataclass
class ParallelDocument:
    """한영 병렬 문서"""
    korean: str
    english: str
    source: str  # 데이터 출처
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return asdict(self)


class GitHubCollector:
    """GitHub에서 한국어 마크다운 문서 수집

    한국어와 영어 버전이 모두 있는 문서를 찾아 수집합니다.
    """

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(self, token: Optional[str] = None):
        """
        Args:
            token: GitHub Personal Access Token (API rate limit 완화)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"

    def _api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """GitHub API 요청"""
        url = f"{self.GITHUB_API_BASE}{endpoint}"
        response = self.session.get(url, params=params)

        if response.status_code == 403:
            # Rate limit
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait_time = max(reset_time - time.time(), 60)
            print(f"Rate limited. Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time)
            return self._api_request(endpoint, params)

        response.raise_for_status()
        return response.json()

    def search_korean_repos(
        self,
        query: str = "README language:Korean",
        max_repos: int = 100
    ) -> List[Dict]:
        """한국어 문서가 있는 레포지토리 검색

        Args:
            query: 검색 쿼리
            max_repos: 최대 레포 수

        Returns:
            레포지토리 정보 리스트
        """
        repos = []
        page = 1
        per_page = min(100, max_repos)

        while len(repos) < max_repos:
            result = self._api_request("/search/repositories", {
                "q": query,
                "per_page": per_page,
                "page": page,
                "sort": "stars"
            })

            items = result.get("items", [])
            if not items:
                break

            repos.extend(items)
            page += 1

            # Rate limit 고려
            time.sleep(1)

        return repos[:max_repos]

    def find_bilingual_docs(
        self,
        owner: str,
        repo: str
    ) -> List[Tuple[str, str]]:
        """레포지토리에서 한영 병렬 문서 찾기

        일반적인 패턴:
        - README.md / README.en.md
        - docs/ko/... / docs/en/...
        - *.ko.md / *.en.md
        """
        pairs = []

        try:
            # 레포 파일 트리 가져오기
            tree = self._api_request(f"/repos/{owner}/{repo}/git/trees/main", {
                "recursive": "1"
            })
        except:
            try:
                tree = self._api_request(f"/repos/{owner}/{repo}/git/trees/master", {
                    "recursive": "1"
                })
            except:
                return pairs

        files = [item["path"] for item in tree.get("tree", []) if item["type"] == "blob"]
        md_files = [f for f in files if f.endswith(".md")]

        # 패턴 1: *.ko.md / *.en.md
        for ko_file in md_files:
            if ".ko.md" in ko_file:
                en_file = ko_file.replace(".ko.md", ".en.md")
                if en_file in md_files:
                    pairs.append((ko_file, en_file))

        # 패턴 2: docs/ko/ / docs/en/
        ko_docs = [f for f in md_files if "/ko/" in f or f.startswith("ko/")]
        for ko_file in ko_docs:
            en_file = ko_file.replace("/ko/", "/en/")
            if en_file in md_files:
                pairs.append((ko_file, en_file))

        return pairs

    def download_file_content(
        self,
        owner: str,
        repo: str,
        path: str
    ) -> Optional[str]:
        """파일 내용 다운로드"""
        try:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
            response = requests.get(url)
            if response.status_code == 404:
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"
                response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Failed to download {path}: {e}")
            return None

    def collect_from_repos(
        self,
        repos: List[Dict],
        output_dir: str = "data/raw/github"
    ) -> List[ParallelDocument]:
        """레포지토리들에서 병렬 문서 수집

        Args:
            repos: 레포지토리 정보 리스트
            output_dir: 저장 디렉토리

        Returns:
            수집된 문서 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        documents = []

        for repo in tqdm(repos, desc="Collecting from repos"):
            owner = repo["owner"]["login"]
            repo_name = repo["name"]

            pairs = self.find_bilingual_docs(owner, repo_name)

            for ko_path, en_path in pairs:
                ko_content = self.download_file_content(owner, repo_name, ko_path)
                en_content = self.download_file_content(owner, repo_name, en_path)

                if ko_content and en_content:
                    doc = ParallelDocument(
                        korean=ko_content,
                        english=en_content,
                        source=f"github:{owner}/{repo_name}",
                        metadata={
                            "ko_path": ko_path,
                            "en_path": en_path,
                            "stars": repo.get("stargazers_count", 0)
                        }
                    )
                    documents.append(doc)

                time.sleep(0.5)  # Rate limit

        # 저장
        output_file = output_path / "github_parallel.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')

        print(f"Collected {len(documents)} documents from GitHub")
        return documents


class AIHubLoader:
    """AI Hub 한영 병렬 코퍼스 로더

    AI Hub에서 다운로드한 데이터를 로드합니다.
    (수동 다운로드 필요: https://aihub.or.kr)
    """

    def __init__(self, data_dir: str = "data/raw/aihub"):
        self.data_dir = Path(data_dir)

    def load_json_corpus(self, file_path: str) -> List[ParallelDocument]:
        """AI Hub JSON 형식 코퍼스 로드

        일반적인 형식:
        {
            "data": [
                {"ko": "한국어 문장", "en": "English sentence"},
                ...
            ]
        }
        """
        documents = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data if isinstance(data, list) else data.get("data", [])

        for item in items:
            ko = item.get("ko") or item.get("korean") or item.get("src")
            en = item.get("en") or item.get("english") or item.get("tgt")

            if ko and en:
                doc = ParallelDocument(
                    korean=ko,
                    english=en,
                    source="aihub",
                    metadata={"file": Path(file_path).name}
                )
                documents.append(doc)

        return documents

    def load_tsv_corpus(self, file_path: str) -> List[ParallelDocument]:
        """TSV 형식 코퍼스 로드 (탭 구분)"""
        documents = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc = ParallelDocument(
                        korean=parts[0],
                        english=parts[1],
                        source="aihub",
                        metadata={"file": Path(file_path).name}
                    )
                    documents.append(doc)

        return documents

    def load_all(self) -> List[ParallelDocument]:
        """데이터 디렉토리의 모든 파일 로드"""
        if not self.data_dir.exists():
            print(f"AI Hub data directory not found: {self.data_dir}")
            return []

        documents = []

        # JSON 파일
        for json_file in self.data_dir.glob("**/*.json"):
            try:
                docs = self.load_json_corpus(str(json_file))
                documents.extend(docs)
                print(f"Loaded {len(docs)} from {json_file.name}")
            except Exception as e:
                print(f"Failed to load {json_file}: {e}")

        # TSV 파일
        for tsv_file in self.data_dir.glob("**/*.tsv"):
            try:
                docs = self.load_tsv_corpus(str(tsv_file))
                documents.extend(docs)
                print(f"Loaded {len(docs)} from {tsv_file.name}")
            except Exception as e:
                print(f"Failed to load {tsv_file}: {e}")

        print(f"Total loaded from AI Hub: {len(documents)}")
        return documents


class OPUSLoader:
    """OPUS 병렬 코퍼스 로더

    OPUS 프로젝트의 한영 병렬 데이터를 로드합니다.
    (https://opus.nlpl.eu/)
    """

    OPUS_CORPORA = {
        "OpenSubtitles": "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-ko.txt.zip",
        "CCMatrix": "https://opus.nlpl.eu/download.php?f=CCMatrix/v1/moses/en-ko.txt.zip",
        "WikiMatrix": "https://opus.nlpl.eu/download.php?f=WikiMatrix/v1/moses/en-ko.txt.zip",
    }

    def __init__(self, data_dir: str = "data/raw/opus"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_corpus(self, corpus_name: str) -> Optional[Path]:
        """코퍼스 다운로드"""
        if corpus_name not in self.OPUS_CORPORA:
            print(f"Unknown corpus: {corpus_name}")
            return None

        url = self.OPUS_CORPORA[corpus_name]
        output_path = self.data_dir / f"{corpus_name}.zip"

        if output_path.exists():
            print(f"Corpus already downloaded: {output_path}")
            return output_path

        print(f"Downloading {corpus_name}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return output_path

    def load_moses_format(
        self,
        ko_file: str,
        en_file: str,
        max_lines: int = None
    ) -> List[ParallelDocument]:
        """Moses 형식 (별도 파일) 로드"""
        documents = []

        with open(ko_file, 'r', encoding='utf-8') as f_ko, \
             open(en_file, 'r', encoding='utf-8') as f_en:

            for i, (ko_line, en_line) in enumerate(zip(f_ko, f_en)):
                if max_lines and i >= max_lines:
                    break

                ko = ko_line.strip()
                en = en_line.strip()

                if ko and en:
                    doc = ParallelDocument(
                        korean=ko,
                        english=en,
                        source="opus",
                        metadata={"line": i}
                    )
                    documents.append(doc)

        return documents


class MarkdownDataFilter:
    """마크다운 데이터 필터

    마크다운 특성을 가진 데이터만 필터링합니다.
    """

    MARKDOWN_INDICATORS = [
        r'^#{1,6}\s+',           # 헤더
        r'\*\*[^*]+\*\*',        # 볼드
        r'\[[^\]]+\]\([^)]+\)',  # 링크
        r'```',                  # 코드 블록
        r'`[^`]+`',              # 인라인 코드
        r'^\s*[-*+]\s+',         # 리스트
        r'^\s*\d+\.\s+',         # 번호 리스트
        r'^\|.+\|$',             # 테이블
    ]

    def __init__(self, min_indicators: int = 2):
        """
        Args:
            min_indicators: 마크다운으로 판단할 최소 지표 수
        """
        self.min_indicators = min_indicators
        self.patterns = [re.compile(p, re.MULTILINE) for p in self.MARKDOWN_INDICATORS]

    def is_markdown(self, text: str) -> bool:
        """마크다운 여부 판단"""
        count = sum(1 for p in self.patterns if p.search(text))
        return count >= self.min_indicators

    def filter_documents(
        self,
        documents: List[ParallelDocument]
    ) -> List[ParallelDocument]:
        """마크다운 문서만 필터링"""
        filtered = []

        for doc in documents:
            # 한국어나 영어 중 하나라도 마크다운이면 포함
            if self.is_markdown(doc.korean) or self.is_markdown(doc.english):
                filtered.append(doc)

        print(f"Filtered: {len(documents)} -> {len(filtered)} markdown documents")
        return filtered


class DataCollector:
    """통합 데이터 수집기

    여러 소스에서 데이터를 수집하고 통합합니다.
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        github_token: str = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.github_collector = GitHubCollector(token=github_token)
        self.aihub_loader = AIHubLoader(str(self.output_dir / "aihub"))
        self.opus_loader = OPUSLoader(str(self.output_dir / "opus"))
        self.md_filter = MarkdownDataFilter()

    def collect_github(
        self,
        query: str = "README language:Korean",
        max_repos: int = 50
    ) -> List[ParallelDocument]:
        """GitHub에서 수집"""
        print("\n=== Collecting from GitHub ===")
        repos = self.github_collector.search_korean_repos(query, max_repos)
        return self.github_collector.collect_from_repos(repos)

    def load_aihub(self) -> List[ParallelDocument]:
        """AI Hub 데이터 로드"""
        print("\n=== Loading AI Hub data ===")
        return self.aihub_loader.load_all()

    def collect_all(
        self,
        include_github: bool = True,
        include_aihub: bool = True,
        filter_markdown: bool = True
    ) -> List[ParallelDocument]:
        """모든 소스에서 수집

        Args:
            include_github: GitHub 수집 포함
            include_aihub: AI Hub 로드 포함
            filter_markdown: 마크다운 필터링 적용

        Returns:
            수집된 문서 리스트
        """
        all_documents = []

        if include_github:
            github_docs = self.collect_github()
            all_documents.extend(github_docs)

        if include_aihub:
            aihub_docs = self.load_aihub()
            all_documents.extend(aihub_docs)

        # 마크다운 필터링
        if filter_markdown:
            all_documents = self.md_filter.filter_documents(all_documents)

        # 중복 제거 (한국어 기준)
        seen = set()
        unique_docs = []
        for doc in all_documents:
            key = doc.korean[:200]  # 첫 200자로 판단
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        print(f"\nTotal unique documents: {len(unique_docs)}")

        # 저장
        output_file = self.output_dir / "collected_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in unique_docs:
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')

        print(f"Saved to {output_file}")
        return unique_docs
