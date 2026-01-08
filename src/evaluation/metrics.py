"""번역 품질 평가 메트릭

BLEU, chrF, COMET 등 번역 품질 평가 메트릭을 제공합니다.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """평가 결과"""
    bleu: float
    chrf: float
    comet: Optional[float] = None
    preservation_rate: float = 0.0
    details: Dict = None


class TranslationMetrics:
    """번역 품질 평가 메트릭

    BLEU, chrF, COMET 스코어를 계산합니다.
    """

    def __init__(self, use_comet: bool = True):
        """
        Args:
            use_comet: COMET 메트릭 사용 여부
        """
        self.use_comet = use_comet
        self._comet_model = None

    def _load_comet_model(self):
        """COMET 모델 로드 (lazy loading)"""
        if self._comet_model is None and self.use_comet:
            try:
                from comet import download_model, load_from_checkpoint
                model_path = download_model("Unbabel/wmt22-comet-da")
                self._comet_model = load_from_checkpoint(model_path)
                print("COMET model loaded successfully")
            except Exception as e:
                print(f"Failed to load COMET model: {e}")
                self.use_comet = False
        return self._comet_model

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """BLEU 스코어 계산

        Args:
            predictions: 예측 번역 리스트
            references: 참조 번역 리스트 (각 예측에 대해 여러 참조 가능)

        Returns:
            BLEU 관련 메트릭
        """
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(predictions, references)

        return {
            "bleu": bleu.score,
            "bleu_bp": bleu.bp,  # Brevity penalty
            "bleu_ratio": bleu.ratio,
            "bleu_1": bleu.precisions[0] if bleu.precisions else 0,
            "bleu_2": bleu.precisions[1] if len(bleu.precisions) > 1 else 0,
            "bleu_3": bleu.precisions[2] if len(bleu.precisions) > 2 else 0,
            "bleu_4": bleu.precisions[3] if len(bleu.precisions) > 3 else 0,
        }

    def compute_chrf(
        self,
        predictions: List[str],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """chrF 스코어 계산

        문자 수준 F-score로, 한국어처럼 형태소가 복잡한 언어에 효과적입니다.

        Args:
            predictions: 예측 번역 리스트
            references: 참조 번역 리스트

        Returns:
            chrF 메트릭
        """
        import sacrebleu

        chrf = sacrebleu.corpus_chrf(predictions, references)

        return {
            "chrf": chrf.score,
            "chrf_precision": chrf.char_prec if hasattr(chrf, 'char_prec') else None,
            "chrf_recall": chrf.char_recall if hasattr(chrf, 'char_recall') else None,
        }

    def compute_comet(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """COMET 스코어 계산

        신경망 기반 품질 평가로, 인간 판단과 높은 상관관계를 보입니다.

        Args:
            sources: 원본 텍스트 (한국어)
            predictions: 예측 번역
            references: 참조 번역

        Returns:
            COMET 메트릭
        """
        model = self._load_comet_model()
        if model is None:
            return {"comet": None, "comet_scores": []}

        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        try:
            output = model.predict(data, batch_size=8, progress_bar=True)
            return {
                "comet": output.system_score,
                "comet_scores": output.scores
            }
        except Exception as e:
            print(f"COMET computation failed: {e}")
            return {"comet": None, "comet_scores": []}

    def evaluate_all(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """모든 메트릭 계산

        Args:
            sources: 원본 텍스트 (한국어)
            predictions: 예측 번역
            references: 참조 번역

        Returns:
            모든 메트릭 결과
        """
        # 참조를 리스트로 래핑 (sacrebleu 형식)
        refs_wrapped = [[ref] for ref in references]

        results = {}

        # BLEU
        bleu_results = self.compute_bleu(predictions, refs_wrapped)
        results.update(bleu_results)

        # chrF
        chrf_results = self.compute_chrf(predictions, refs_wrapped)
        results.update(chrf_results)

        # COMET
        if self.use_comet:
            comet_results = self.compute_comet(sources, predictions, references)
            results.update(comet_results)

        return results


class MarkdownPreservationMetrics:
    """마크다운 구조 보존 평가

    번역 후 마크다운 요소가 얼마나 잘 보존되었는지 평가합니다.
    """

    MARKDOWN_PATTERNS = {
        'headers': re.compile(r'^#{1,6}\s+', re.MULTILINE),
        'code_blocks': re.compile(r'```[\s\S]*?```'),
        'inline_code': re.compile(r'`[^`\n]+`'),
        'links': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
        'images': re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
        'bold': re.compile(r'\*\*[^*]+\*\*'),
        'italic': re.compile(r'(?<!\*)\*[^*]+\*(?!\*)'),
        'tables': re.compile(r'^\|.+\|$', re.MULTILINE),
        'lists': re.compile(r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+', re.MULTILINE),
        'blockquotes': re.compile(r'^>\s+', re.MULTILINE),
    }

    def count_elements(self, text: str) -> Dict[str, int]:
        """마크다운 요소 개수 계산

        Args:
            text: 마크다운 텍스트

        Returns:
            요소별 개수
        """
        counts = {}
        for name, pattern in self.MARKDOWN_PATTERNS.items():
            matches = pattern.findall(text)
            counts[name] = len(matches)
        return counts

    def compute_preservation_rate(
        self,
        source: str,
        translation: str
    ) -> Dict[str, float]:
        """마크다운 보존율 계산

        Args:
            source: 원본 마크다운
            translation: 번역된 마크다운

        Returns:
            요소별 보존율
        """
        source_counts = self.count_elements(source)
        trans_counts = self.count_elements(translation)

        preservation_rates = {}

        for element in self.MARKDOWN_PATTERNS.keys():
            src_count = source_counts[element]
            trans_count = trans_counts[element]

            if src_count > 0:
                # 보존율은 1을 초과할 수 없음
                rate = min(trans_count / src_count, 1.0)
            else:
                # 원본에 없으면 1.0 (번역에도 없어야 정상)
                rate = 1.0 if trans_count == 0 else 0.5

            preservation_rates[f'{element}_rate'] = rate

        # 전체 보존율 계산
        total_src = sum(source_counts.values())
        total_trans = sum(trans_counts.values())

        if total_src > 0:
            preservation_rates['overall_rate'] = min(total_trans / total_src, 1.0)
        else:
            preservation_rates['overall_rate'] = 1.0

        preservation_rates['source_counts'] = source_counts
        preservation_rates['translation_counts'] = trans_counts

        return preservation_rates

    def evaluate_batch(
        self,
        sources: List[str],
        translations: List[str]
    ) -> Dict[str, float]:
        """배치 평가

        Args:
            sources: 원본 마크다운 리스트
            translations: 번역된 마크다운 리스트

        Returns:
            평균 보존율
        """
        all_rates = []

        for src, trans in zip(sources, translations):
            rates = self.compute_preservation_rate(src, trans)
            all_rates.append(rates)

        # 평균 계산
        avg_rates = {}
        rate_keys = [k for k in all_rates[0].keys() if k.endswith('_rate')]

        for key in rate_keys:
            values = [r[key] for r in all_rates]
            avg_rates[f'avg_{key}'] = sum(values) / len(values)

        return avg_rates

    def find_missing_elements(
        self,
        source: str,
        translation: str
    ) -> Dict[str, List[str]]:
        """누락된 요소 찾기

        Args:
            source: 원본 마크다운
            translation: 번역된 마크다운

        Returns:
            누락된 요소 정보
        """
        missing = {}

        # 코드 블록 확인
        src_code_blocks = self.MARKDOWN_PATTERNS['code_blocks'].findall(source)
        trans_code_blocks = self.MARKDOWN_PATTERNS['code_blocks'].findall(translation)

        if len(src_code_blocks) > len(trans_code_blocks):
            missing['code_blocks'] = {
                'source_count': len(src_code_blocks),
                'translation_count': len(trans_code_blocks),
                'missing_count': len(src_code_blocks) - len(trans_code_blocks)
            }

        # 링크 확인
        src_links = self.MARKDOWN_PATTERNS['links'].findall(source)
        trans_links = self.MARKDOWN_PATTERNS['links'].findall(translation)

        if len(src_links) > len(trans_links):
            missing['links'] = {
                'source_count': len(src_links),
                'translation_count': len(trans_links),
                'source_urls': [url for _, url in src_links],
                'translation_urls': [url for _, url in trans_links]
            }

        return missing


class CombinedEvaluator:
    """통합 평가기

    번역 품질과 마크다운 보존율을 함께 평가합니다.
    """

    def __init__(self, use_comet: bool = True):
        self.translation_metrics = TranslationMetrics(use_comet=use_comet)
        self.preservation_metrics = MarkdownPreservationMetrics()

    def evaluate(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str]
    ) -> EvaluationResult:
        """통합 평가

        Args:
            sources: 원본 한국어 텍스트
            predictions: 예측 번역
            references: 참조 번역

        Returns:
            EvaluationResult
        """
        # 번역 품질 평가
        translation_results = self.translation_metrics.evaluate_all(
            sources, predictions, references
        )

        # 마크다운 보존율 평가
        preservation_results = self.preservation_metrics.evaluate_batch(
            sources, predictions
        )

        # 결과 통합
        return EvaluationResult(
            bleu=translation_results.get('bleu', 0),
            chrf=translation_results.get('chrf', 0),
            comet=translation_results.get('comet'),
            preservation_rate=preservation_results.get('avg_overall_rate', 0),
            details={
                'translation': translation_results,
                'preservation': preservation_results
            }
        )

    def print_report(self, result: EvaluationResult):
        """평가 결과 출력"""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print("\n[Translation Quality]")
        print(f"  BLEU:  {result.bleu:.2f}")
        print(f"  chrF:  {result.chrf:.2f}")
        if result.comet is not None:
            print(f"  COMET: {result.comet:.4f}")

        print("\n[Markdown Preservation]")
        print(f"  Overall Rate: {result.preservation_rate:.2%}")

        if result.details and 'preservation' in result.details:
            pres = result.details['preservation']
            for key, value in pres.items():
                if key.startswith('avg_') and key.endswith('_rate'):
                    element = key[4:-5]  # Remove 'avg_' and '_rate'
                    print(f"  {element}: {value:.2%}")

        print("\n" + "=" * 60)
