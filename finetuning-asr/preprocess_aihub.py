#!/usr/bin/env python
"""
AI Hub 데이터 → VibeVoice 학습 형식 전처리 스크립트

AI Hub에서 다운로드한 음성 데이터를 VibeVoice ASR 학습 형식으로 변환합니다.
주소 관련 발화 필터링 기능 포함.

사용법:
    # 기본 변환
    python preprocess_aihub.py --input-dir /data/aihub/dataset --output-dir /data/processed

    # 주소 발화만 필터링
    python preprocess_aihub.py --input-dir /data/aihub/dataset --output-dir /data/processed --filter-address

    # 긴급 상황 신고 형식
    python preprocess_aihub.py --input-dir /data/aihub/emergency --output-dir /data/processed --format emergency

    # 변환 후 파이프라인 임포트
    python demo/realtime_asr/pipeline.py import --data-dir /data/processed

지원 형식:
    --format aihub_default   : AI Hub 일반 형식 (utterance 기반)
    --format aihub_emergency : 긴급 상황 신고 형식
    --format ksponspeech     : KsponSpeech 형식
    --format custom          : 커스텀 (audio + text 파일 쌍)
"""

import argparse
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("preprocess_aihub")


# 주소 필터링 패턴
ADDRESS_PATTERNS = [
    r"[가-힣]+(?:특별시|광역시|도)\s",
    r"[가-힣]+[시군구]\s",
    r"[가-힣]+(?:로|대로|길)\s*\d+",
    r"[가-힣]+[동리읍면]\s*\d+",
    r"\d+번지",
    r"[가-힣]+사거리",
    r"[가-힣]+교차로",
    r"[가-힣]+아파트",
    r"[가-힣]+IC",
    r"[가-힣]+고속도로",
    r"\d+동\s*\d+호",
    r"[가-힣]+[역]\s*\d+번\s*출구",
]

# 핫워드 자동 추출용 키워드
ADDRESS_HOTWORD_TRIGGERS = [
    "시", "구", "군", "동", "읍", "면", "리",
    "로", "대로", "길", "번길",
    "아파트", "빌딩", "타워", "센터", "프라자", "몰",
    "사거리", "삼거리", "교차로",
    "IC", "JC", "고속도로", "국도",
    "역", "출구", "정류장", "터미널",
]


def contains_address(text: str) -> bool:
    """텍스트에 주소 패턴이 포함되어 있는지 확인"""
    for pattern in ADDRESS_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def extract_hotwords_from_text(text: str) -> List[str]:
    """텍스트에서 주소 관련 핫워드를 자동 추출"""
    hotwords = []

    # 시도
    for m in re.finditer(r"([가-힣]+(?:특별시|광역시|도|시|군|구))", text):
        hotwords.append(m.group(1))

    # 도로명
    for m in re.finditer(r"([가-힣]+(?:로|대로|길))", text):
        hotwords.append(m.group(1))

    # 건물명
    for m in re.finditer(r"([가-힣]+(?:아파트|빌딩|타워|센터|프라자|병원|학교|시장))", text):
        hotwords.append(m.group(1))

    # 교차로
    for m in re.finditer(r"([가-힣]+(?:사거리|삼거리|교차로))", text):
        hotwords.append(m.group(1))

    # IC/JC
    for m in re.finditer(r"([가-힣]+(?:IC|JC))", text):
        hotwords.append(m.group(1))

    return list(set(hotwords))


class AIHubPreprocessor:
    """AI Hub 데이터 전처리기"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        format_type: str = "aihub_default",
        filter_address: bool = False,
        max_duration: float = 600.0,
        min_duration: float = 1.0,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        self.filter_address = filter_address
        self.max_duration = max_duration
        self.min_duration = min_duration

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total": 0, "converted": 0, "filtered": 0,
            "skipped": 0, "errors": 0,
        }

    def process(self):
        """전체 데이터셋을 처리합니다."""
        logger.info(f"입력: {self.input_dir}")
        logger.info(f"출력: {self.output_dir}")
        logger.info(f"형식: {self.format_type}")
        logger.info(f"주소 필터: {'ON' if self.filter_address else 'OFF'}")

        if self.format_type == "aihub_default":
            self._process_aihub_default()
        elif self.format_type == "aihub_emergency":
            self._process_aihub_emergency()
        elif self.format_type == "ksponspeech":
            self._process_ksponspeech()
        elif self.format_type == "custom":
            self._process_custom()
        else:
            logger.error(f"알 수 없는 형식: {self.format_type}")
            return

        logger.info("=" * 50)
        logger.info(f"전처리 완료:")
        logger.info(f"  전체: {self.stats['total']}건")
        logger.info(f"  변환: {self.stats['converted']}건")
        if self.filter_address:
            logger.info(f"  주소 필터 제외: {self.stats['filtered']}건")
        logger.info(f"  스킵: {self.stats['skipped']}건")
        logger.info(f"  오류: {self.stats['errors']}건")

    def _process_aihub_default(self):
        """AI Hub 일반 형식 처리

        구조:
          라벨링데이터/*.json + 원천데이터/*.wav
        """
        # JSON 파일 탐색
        json_dirs = list(self.input_dir.rglob("라벨링데이터"))
        audio_dirs = list(self.input_dir.rglob("원천데이터"))

        if not json_dirs:
            # 플랫 구조 시도
            json_files = list(self.input_dir.rglob("*.json"))
        else:
            json_files = []
            for jd in json_dirs:
                json_files.extend(jd.rglob("*.json"))

        logger.info(f"JSON 파일 {len(json_files)}개 발견")

        for json_path in sorted(json_files):
            self.stats["total"] += 1
            try:
                self._convert_aihub_json(json_path)
            except Exception as e:
                self.stats["errors"] += 1
                if self.stats["errors"] <= 10:
                    logger.warning(f"변환 오류 [{json_path.name}]: {e}")

            if self.stats["total"] % 500 == 0:
                logger.info(f"진행: {self.stats['total']}건 (변환 {self.stats['converted']})")

    def _convert_aihub_json(self, json_path: Path):
        """AI Hub JSON 한 건을 VibeVoice 형식으로 변환"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 세그먼트 추출
        segments = []
        utterances = data.get("utterance", data.get("utterances", []))

        if not utterances:
            # 단일 텍스트 형식
            text = data.get("text", data.get("transcription", ""))
            if text:
                segments = [{"speaker": 0, "text": text, "start": 0.0, "end": 0.0}]
        else:
            for utt in utterances:
                seg = {
                    "speaker": utt.get("speaker_id", utt.get("speaker", 0)),
                    "text": utt.get("form", utt.get("text", utt.get("sentence", ""))),
                    "start": round(float(utt.get("start", utt.get("start_time", 0))), 2),
                    "end": round(float(utt.get("end", utt.get("end_time", 0))), 2),
                }
                if isinstance(seg["speaker"], str):
                    # speaker ID를 정수로 변환
                    seg["speaker"] = hash(seg["speaker"]) % 10
                segments.append(seg)

        if not segments:
            self.stats["skipped"] += 1
            return

        # 전체 텍스트
        full_text = " ".join(seg["text"] for seg in segments)

        # 주소 필터링
        if self.filter_address and not contains_address(full_text):
            self.stats["filtered"] += 1
            return

        # 오디오 파일 찾기
        audio_path = self._find_audio(json_path, data)
        if not audio_path:
            self.stats["skipped"] += 1
            return

        # 길이 확인
        duration = data.get("audio_duration",
                           data.get("duration",
                           data.get("metadata", {}).get("duration", 0)))

        if duration and (duration < self.min_duration or duration > self.max_duration):
            self.stats["skipped"] += 1
            return

        # 핫워드 추출
        hotwords = extract_hotwords_from_text(full_text)

        # 출력 파일 생성
        out_name = json_path.stem
        out_json = self.output_dir / f"{out_name}.json"
        out_audio = self.output_dir / f"{out_name}{audio_path.suffix}"

        if out_json.exists():
            self.stats["skipped"] += 1
            return

        # 오디오 복사
        shutil.copy2(audio_path, out_audio)

        # VibeVoice 형식 JSON 저장
        vibevoice_data = {
            "audio_path": out_audio.name,
            "audio_duration": round(duration, 2) if duration else 0,
            "segments": segments,
            "customized_context": hotwords,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)

        self.stats["converted"] += 1

    def _process_aihub_emergency(self):
        """긴급 상황 신고 음성 형식 처리"""
        # 긴급 상황 데이터는 보통 대화 형식
        for json_path in sorted(self.input_dir.rglob("*.json")):
            self.stats["total"] += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 대화 턴 추출
                turns = data.get("dialog", data.get("turns", data.get("conversation", [])))
                segments = []

                for turn in turns:
                    role = turn.get("role", turn.get("speaker", ""))
                    # caller=0, dispatcher=1
                    speaker = 0 if role in ("caller", "신고자", "user", "발화자") else 1
                    segments.append({
                        "speaker": speaker,
                        "text": turn.get("text", turn.get("utterance", turn.get("content", ""))),
                        "start": round(float(turn.get("start", 0)), 2),
                        "end": round(float(turn.get("end", 0)), 2),
                    })

                if not segments:
                    self.stats["skipped"] += 1
                    continue

                full_text = " ".join(s["text"] for s in segments)
                if self.filter_address and not contains_address(full_text):
                    self.stats["filtered"] += 1
                    continue

                audio_path = self._find_audio(json_path, data)
                if not audio_path:
                    self.stats["skipped"] += 1
                    continue

                hotwords = extract_hotwords_from_text(full_text)
                # 긴급 키워드 추가
                emergency_kw = ["화재", "구급", "구조", "사고", "119"]
                for kw in emergency_kw:
                    if kw in full_text:
                        hotwords.append(kw)

                out_name = json_path.stem
                out_json = self.output_dir / f"{out_name}.json"
                out_audio = self.output_dir / f"{out_name}{audio_path.suffix}"

                if not out_json.exists():
                    shutil.copy2(audio_path, out_audio)
                    vibevoice_data = {
                        "audio_path": out_audio.name,
                        "audio_duration": data.get("duration", 0),
                        "segments": segments,
                        "customized_context": list(set(hotwords)),
                    }
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)
                    self.stats["converted"] += 1

            except Exception as e:
                self.stats["errors"] += 1
                if self.stats["errors"] <= 10:
                    logger.warning(f"변환 오류: {e}")

    def _process_ksponspeech(self):
        """KsponSpeech 형식 처리

        구조: *.pcm 또는 *.wav + *.txt (탭 구분 전사)
        """
        for txt_path in sorted(self.input_dir.rglob("*.txt")):
            self.stats["total"] += 1
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        parts = line.strip().split(" :: ")
                    if len(parts) < 2:
                        continue

                    audio_name = parts[0].strip()
                    text = parts[1].strip()

                    # 괄호/태그 제거
                    text = re.sub(r"\([^)]*\)", "", text)
                    text = re.sub(r"[/+*]", "", text)
                    text = text.strip()

                    if not text or len(text) < 3:
                        continue

                    if self.filter_address and not contains_address(text):
                        self.stats["filtered"] += 1
                        continue

                    # 오디오 찾기
                    audio_path = None
                    for ext in [".wav", ".pcm", ".flac", ".mp3"]:
                        candidate = txt_path.parent / f"{audio_name}{ext}"
                        if candidate.exists():
                            audio_path = candidate
                            break
                    if not audio_path:
                        for ext in [".wav", ".pcm", ".flac", ".mp3"]:
                            candidates = list(self.input_dir.rglob(f"{audio_name}{ext}"))
                            if candidates:
                                audio_path = candidates[0]
                                break

                    if not audio_path:
                        self.stats["skipped"] += 1
                        continue

                    hotwords = extract_hotwords_from_text(text)
                    out_name = audio_name.replace("/", "_").replace("\\", "_")
                    out_json = self.output_dir / f"{out_name}.json"
                    out_audio = self.output_dir / f"{out_name}{audio_path.suffix}"

                    if not out_json.exists():
                        shutil.copy2(audio_path, out_audio)
                        vibevoice_data = {
                            "audio_path": out_audio.name,
                            "audio_duration": 0,
                            "segments": [{"speaker": 0, "text": text, "start": 0.0, "end": 0.0}],
                            "customized_context": hotwords,
                        }
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)
                        self.stats["converted"] += 1

            except Exception as e:
                self.stats["errors"] += 1

    def _process_custom(self):
        """커스텀 형식: audio + json 1:1 매칭"""
        for audio_path in sorted(self.input_dir.rglob("*.wav")):
            self.stats["total"] += 1
            json_path = audio_path.with_suffix(".json")
            txt_path = audio_path.with_suffix(".txt")

            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 이미 VibeVoice 형식이면 그대로 복사
                    if "segments" in data:
                        full_text = " ".join(s.get("text", "") for s in data["segments"])
                        if self.filter_address and not contains_address(full_text):
                            self.stats["filtered"] += 1
                            continue
                        shutil.copy2(audio_path, self.output_dir / audio_path.name)
                        shutil.copy2(json_path, self.output_dir / json_path.name)
                        self.stats["converted"] += 1
                except Exception:
                    self.stats["errors"] += 1

            elif txt_path.exists():
                try:
                    text = txt_path.read_text(encoding="utf-8").strip()
                    if self.filter_address and not contains_address(text):
                        self.stats["filtered"] += 1
                        continue
                    hotwords = extract_hotwords_from_text(text)
                    out_name = audio_path.stem
                    shutil.copy2(audio_path, self.output_dir / audio_path.name)
                    vibevoice_data = {
                        "audio_path": audio_path.name,
                        "audio_duration": 0,
                        "segments": [{"speaker": 0, "text": text, "start": 0.0, "end": 0.0}],
                        "customized_context": hotwords,
                    }
                    with open(self.output_dir / f"{out_name}.json", "w", encoding="utf-8") as f:
                        json.dump(vibevoice_data, f, ensure_ascii=False, indent=2)
                    self.stats["converted"] += 1
                except Exception:
                    self.stats["errors"] += 1

    def _find_audio(self, json_path: Path, data: Dict) -> Optional[Path]:
        """JSON에 대응하는 오디오 파일을 찾습니다."""
        # 1. JSON 내 audio_path 필드
        audio_name = data.get("audio_path", data.get("file_name", data.get("audio_file", "")))
        if audio_name:
            # 같은 디렉토리
            candidate = json_path.parent / audio_name
            if candidate.exists():
                return candidate
            # 원천데이터 디렉토리
            for audio_dir in self.input_dir.rglob("원천데이터"):
                candidate = audio_dir / audio_name
                if candidate.exists():
                    return candidate

        # 2. 같은 이름 다른 확장자
        for ext in [".wav", ".mp3", ".flac", ".pcm", ".m4a"]:
            candidate = json_path.with_suffix(ext)
            if candidate.exists():
                return candidate

        # 3. 원천데이터 디렉토리에서 같은 이름
        stem = json_path.stem
        for ext in [".wav", ".mp3", ".flac"]:
            candidates = list(self.input_dir.rglob(f"{stem}{ext}"))
            if candidates:
                return candidates[0]

        return None


def main():
    parser = argparse.ArgumentParser(description="AI Hub 데이터 → VibeVoice 학습 형식 전처리")
    parser.add_argument("--input-dir", required=True, help="AI Hub 데이터 디렉토리")
    parser.add_argument("--output-dir", required=True, help="출력 디렉토리")
    parser.add_argument("--format", default="aihub_default",
                       choices=["aihub_default", "aihub_emergency", "ksponspeech", "custom"],
                       help="입력 데이터 형식")
    parser.add_argument("--filter-address", action="store_true",
                       help="주소 관련 발화만 필터링")
    parser.add_argument("--max-duration", type=float, default=600.0, help="최대 오디오 길이 (초)")
    parser.add_argument("--min-duration", type=float, default=1.0, help="최소 오디오 길이 (초)")

    args = parser.parse_args()

    processor = AIHubPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        format_type=args.format,
        filter_address=args.filter_address,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )
    processor.process()


if __name__ == "__main__":
    main()
