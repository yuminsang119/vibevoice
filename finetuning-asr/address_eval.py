#!/usr/bin/env python
"""
119 ASR 주소 인식 전용 평가 도구

주소 유형별(도로명/지번/교차로/랜드마크/구어체/고속도로) WER을 세분화하여 측정하고,
주소 구성요소별(시도/구군/도로명/번지/건물명/동호수) 정확도를 분석합니다.

사용법:
    # 기본 평가
    python address_eval.py --model-path microsoft/VibeVoice-ASR --eval-dir ./eval_data

    # 상세 리포트 출력
    python address_eval.py --model-path ./models/119_asr_v0005 --eval-dir ./eval_data --report

    # JSON 결과 저장
    python address_eval.py --model-path ./models/v0005 --eval-dir ./eval_data --output results.json
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("address_eval")

SCRIPT_DIR = Path(__file__).parent


# ---------- 주소 추출 패턴 ----------

ADDRESS_EXTRACTORS = {
    "시도": re.compile(r"([가-힣]+(?:특별시|광역시|특별자치시|특별자치도|도))"),
    "시군구": re.compile(r"([가-힣]+[시군구])"),
    "도로명": re.compile(r"([가-힣]+(?:로|대로|길))\s*(\d+)"),
    "번지": re.compile(r"([가-힣]+[동리읍면])\s*(\d+(?:-\d+)?)번?지?"),
    "건물명": re.compile(r"([가-힣A-Za-z0-9]+(?:아파트|빌딩|타워|센터|프라자|몰|병원|학교|시장|백화점|마트))"),
    "동호수": re.compile(r"(\d+동)\s*(\d+호)"),
    "층수": re.compile(r"(지하\s*\d+층|\d+층|옥상|반지하)"),
    "교차로": re.compile(r"([가-힣]+(?:사거리|삼거리|오거리|교차로))"),
    "고속도로": re.compile(r"([가-힣]+고속도로)"),
    "IC_JC": re.compile(r"([가-힣]+(?:IC|JC|나들목|분기점))"),
}

# 주소 유형 분류
ADDRESS_TYPE_PATTERNS = {
    "road": re.compile(r"[가-힣]+(?:로|대로|길)\s*\d+"),
    "jibeon": re.compile(r"[가-힣]+[동리읍면]\s*\d+"),
    "intersection": re.compile(r"[가-힣]+(?:사거리|삼거리|교차로)"),
    "highway": re.compile(r"[가-힣]+고속도로"),
    "landmark": re.compile(r"(?:아파트|빌딩|타워|센터|백화점|마트|공원|시장|역)"),
}


# ---------- WER/CER 계산 ----------

def compute_wer(ref: str, hyp: str) -> float:
    """단어 오류율 (WER) 계산"""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(ref: str, hyp: str) -> float:
    """문자 오류율 (CER) 계산"""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    d = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


# ---------- 주소 구성요소 분석 ----------

def extract_address_components(text: str) -> Dict[str, List[str]]:
    """텍스트에서 주소 구성요소를 추출합니다."""
    components = {}
    for name, pattern in ADDRESS_EXTRACTORS.items():
        matches = pattern.findall(text)
        if matches:
            # 튜플 결과를 문자열로
            flat = []
            for m in matches:
                if isinstance(m, tuple):
                    flat.append(" ".join(m))
                else:
                    flat.append(m)
            components[name] = flat
    return components


def classify_address_type(text: str) -> str:
    """텍스트의 주소 유형을 분류합니다."""
    for addr_type, pattern in ADDRESS_TYPE_PATTERNS.items():
        if pattern.search(text):
            return addr_type
    return "unknown"


def compute_component_accuracy(
    ref_components: Dict[str, List[str]],
    hyp_components: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """구성요소별 정확도를 계산합니다."""
    result = {}

    for comp_name in ADDRESS_EXTRACTORS.keys():
        ref_items = set(ref_components.get(comp_name, []))
        hyp_items = set(hyp_components.get(comp_name, []))

        if not ref_items:
            continue

        # 정밀도, 재현율, F1
        if hyp_items:
            correct = ref_items & hyp_items
            precision = len(correct) / len(hyp_items)
            recall = len(correct) / len(ref_items)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        # 문자열 유사도 (정확히 일치하지 않더라도 비슷한 경우)
        char_accuracy = 0.0
        if ref_items and hyp_items:
            ref_str = " ".join(sorted(ref_items))
            hyp_str = " ".join(sorted(hyp_items))
            char_accuracy = 1.0 - compute_cer(ref_str, hyp_str)

        result[comp_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "char_accuracy": round(max(0, char_accuracy), 4),
            "ref_count": len(ref_items),
            "hyp_count": len(hyp_items),
        }

    return result


# ---------- 평가기 ----------

class AddressEvaluator:
    """주소 인식 전용 평가기"""

    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.asr_service = None

    def _load_model(self):
        """ASR 모델을 로드합니다."""
        if self.asr_service is not None:
            return

        import sys
        sys.path.insert(0, str(SCRIPT_DIR.parent / "demo" / "realtime_asr"))
        from app import RealtimeASRService

        self.asr_service = RealtimeASRService(model_path=self.model_path, device=self.device)
        self.asr_service.load()

    def evaluate(
        self,
        eval_dir: str,
        use_model: bool = True,
    ) -> Dict[str, Any]:
        """평가를 실행합니다.

        Args:
            eval_dir: 평가 데이터 디렉토리
            use_model: True면 모델 추론 실행, False면 JSON 기반 텍스트 비교만
        """
        eval_path = Path(eval_dir)

        if use_model and self.model_path:
            self._load_model()

        # 평가 데이터 수집
        samples = []
        for json_path in sorted(eval_path.rglob("*.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "segments" not in data:
                    continue

                audio_path = json_path.parent / data.get("audio_path", "")
                ref_text = " ".join(
                    seg.get("text", seg.get("Content", ""))
                    for seg in data["segments"]
                )

                addr_type = data.get("metadata", {}).get("addr_type", "")
                if not addr_type:
                    addr_type = classify_address_type(ref_text)

                sample = {
                    "id": json_path.stem,
                    "audio_path": str(audio_path),
                    "ref_text": ref_text,
                    "addr_type": addr_type,
                    "hotwords": data.get("customized_context", []),
                    "has_audio": audio_path.exists(),
                }
                samples.append(sample)

            except Exception as e:
                logger.warning(f"데이터 로드 오류: {json_path.name}: {e}")

        if not samples:
            logger.warning("평가 데이터가 없습니다.")
            return {}

        logger.info(f"평가 데이터: {len(samples)}건")

        # 유형별 집계
        type_results = {}
        component_totals = {}
        overall_wer = 0.0
        overall_cer = 0.0
        overall_addr_wer = 0.0
        addr_count = 0
        count = 0

        for sample in samples:
            ref_text = sample["ref_text"]

            # 추론 또는 참조 텍스트 사용
            if use_model and self.asr_service and sample["has_audio"]:
                try:
                    if sample["hotwords"]:
                        self.asr_service.set_hotwords(sample["hotwords"])
                    result = self.asr_service.transcribe(sample["audio_path"])
                    hyp_segments = result.get("segments", [])
                    hyp_text = " ".join(
                        seg.get("Content", seg.get("text", ""))
                        for seg in hyp_segments
                    )
                except Exception as e:
                    logger.warning(f"추론 오류: {sample['id']}: {e}")
                    continue
            else:
                # 모델 없이 텍스트 비교 (시뮬레이션)
                hyp_text = ref_text  # 실제로는 모델이 필요
                continue

            # WER/CER 계산
            wer = compute_wer(ref_text, hyp_text)
            cer = compute_cer(ref_text, hyp_text)
            overall_wer += wer
            overall_cer += cer
            count += 1

            # 주소 부분 WER
            ref_components = extract_address_components(ref_text)
            hyp_components = extract_address_components(hyp_text)

            ref_addr = " ".join(" ".join(v) for v in ref_components.values())
            hyp_addr = " ".join(" ".join(v) for v in hyp_components.values())

            if ref_addr.strip():
                addr_wer = compute_wer(ref_addr, hyp_addr)
                overall_addr_wer += addr_wer
                addr_count += 1

            # 유형별 집계
            addr_type = sample["addr_type"]
            if addr_type not in type_results:
                type_results[addr_type] = {"wer_sum": 0, "cer_sum": 0, "addr_wer_sum": 0, "count": 0, "addr_count": 0}

            type_results[addr_type]["wer_sum"] += wer
            type_results[addr_type]["cer_sum"] += cer
            type_results[addr_type]["count"] += 1
            if ref_addr.strip():
                type_results[addr_type]["addr_wer_sum"] += addr_wer
                type_results[addr_type]["addr_count"] += 1

            # 구성요소별 정확도
            comp_acc = compute_component_accuracy(ref_components, hyp_components)
            for comp_name, metrics in comp_acc.items():
                if comp_name not in component_totals:
                    component_totals[comp_name] = {"f1_sum": 0, "char_acc_sum": 0, "count": 0}
                component_totals[comp_name]["f1_sum"] += metrics["f1"]
                component_totals[comp_name]["char_acc_sum"] += metrics["char_accuracy"]
                component_totals[comp_name]["count"] += 1

        # 결과 정리
        result = {
            "model_path": self.model_path,
            "eval_dir": eval_dir,
            "evaluated_at": datetime.now().isoformat(),
            "total_samples": len(samples),
            "evaluated_samples": count,
            "overall": {
                "wer": round(overall_wer / max(count, 1) * 100, 2),
                "cer": round(overall_cer / max(count, 1) * 100, 2),
                "address_wer": round(overall_addr_wer / max(addr_count, 1) * 100, 2),
            },
            "by_type": {},
            "by_component": {},
        }

        for addr_type, tr in type_results.items():
            c = tr["count"]
            ac = tr["addr_count"]
            result["by_type"][addr_type] = {
                "wer": round(tr["wer_sum"] / max(c, 1) * 100, 2),
                "cer": round(tr["cer_sum"] / max(c, 1) * 100, 2),
                "address_wer": round(tr["addr_wer_sum"] / max(ac, 1) * 100, 2),
                "count": c,
            }

        for comp_name, ct in component_totals.items():
            c = ct["count"]
            result["by_component"][comp_name] = {
                "avg_f1": round(ct["f1_sum"] / max(c, 1), 4),
                "avg_char_accuracy": round(ct["char_acc_sum"] / max(c, 1), 4),
                "count": c,
            }

        return result

    @staticmethod
    def print_report(result: Dict[str, Any]):
        """평가 결과를 테이블로 출력합니다."""
        if not result:
            print("평가 결과가 없습니다.")
            return

        print()
        print("=" * 75)
        print("  119 ASR 주소 인식 평가 리포트")
        print("=" * 75)
        print(f"  모델: {result.get('model_path', '-')}")
        print(f"  평가 데이터: {result.get('total_samples', 0)}건 (실제 평가: {result.get('evaluated_samples', 0)}건)")
        print(f"  평가 일시: {result.get('evaluated_at', '-')}")
        print()

        # 전체 결과
        overall = result.get("overall", {})
        print("  [전체 결과]")
        print(f"    전체 WER:  {overall.get('wer', -1):>6.2f}%")
        print(f"    전체 CER:  {overall.get('cer', -1):>6.2f}%")
        print(f"    주소 WER:  {overall.get('address_wer', -1):>6.2f}%")
        print()

        # 유형별 결과
        by_type = result.get("by_type", {})
        if by_type:
            print("  [주소 유형별 WER]")
            print(f"    {'유형':<15} {'WER':>7} {'CER':>7} {'주소WER':>8} {'건수':>5}")
            print("    " + "-" * 45)

            type_names = {
                "road": "도로명주소",
                "jibeon": "지번주소",
                "intersection": "교차로",
                "landmark": "랜드마크",
                "colloquial": "구어체",
                "highway": "고속도로",
                "apartment": "아파트",
                "unknown": "기타",
            }

            for addr_type, metrics in sorted(by_type.items()):
                name = type_names.get(addr_type, addr_type)
                print(
                    f"    {name:<15} "
                    f"{metrics['wer']:>6.2f}% "
                    f"{metrics['cer']:>6.2f}% "
                    f"{metrics['address_wer']:>7.2f}% "
                    f"{metrics['count']:>5}"
                )
            print()

        # 구성요소별 결과
        by_comp = result.get("by_component", {})
        if by_comp:
            print("  [주소 구성요소별 정확도]")
            print(f"    {'구성요소':<12} {'F1':>8} {'문자정확도':>10} {'건수':>5}")
            print("    " + "-" * 38)

            comp_names = {
                "시도": "시도",
                "시군구": "시군구",
                "도로명": "도로명",
                "번지": "번지",
                "건물명": "건물명",
                "동호수": "동호수",
                "층수": "층수",
                "교차로": "교차로",
                "고속도로": "고속도로",
                "IC_JC": "IC/JC",
            }

            for comp_name, metrics in by_comp.items():
                name = comp_names.get(comp_name, comp_name)
                print(
                    f"    {name:<12} "
                    f"{metrics['avg_f1']:>7.2%} "
                    f"{metrics['avg_char_accuracy']:>9.2%} "
                    f"{metrics['count']:>5}"
                )
            print()

        print("=" * 75)
        print()


def main():
    parser = argparse.ArgumentParser(description="119 ASR 주소 인식 평가")
    parser.add_argument("--model-path", help="ASR 모델 경로")
    parser.add_argument("--eval-dir", required=True, help="평가 데이터 디렉토리")
    parser.add_argument("--device", default="cuda", help="디바이스")
    parser.add_argument("--report", action="store_true", help="상세 리포트 출력")
    parser.add_argument("--output", help="결과 JSON 저장 경로")

    args = parser.parse_args()

    evaluator = AddressEvaluator(model_path=args.model_path, device=args.device)
    result = evaluator.evaluate(
        eval_dir=args.eval_dir,
        use_model=bool(args.model_path),
    )

    if args.report or not args.output:
        AddressEvaluator.print_report(result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"결과 저장: {args.output}")


if __name__ == "__main__":
    main()
