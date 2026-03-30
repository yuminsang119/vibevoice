"""
119 신고전화 지속 학습 파이프라인

실시간 통화 수집 → 데이터 검증 → LoRA 학습 → 평가 → 배포
WER을 지속적으로 낮추는 자동화 파이프라인입니다.

사용법:
    python pipeline.py collect   # 통화 데이터 수집 시작
    python pipeline.py validate  # 수집 데이터 검증
    python pipeline.py train     # LoRA 학습 실행
    python pipeline.py evaluate  # WER 평가
    python pipeline.py deploy    # 모델 배포
    python pipeline.py run       # 전체 파이프라인 자동 실행
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("pipeline")

# ---------- 경로 설정 ----------

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent.parent
FINETUNE_DIR = REPO_ROOT / "finetuning-asr"

# 파이프라인 데이터 경로
PIPELINE_DATA_DIR = BASE_DIR / "pipeline_data"
RAW_DIR = PIPELINE_DATA_DIR / "raw"            # 원본 통화 녹음
PENDING_DIR = PIPELINE_DATA_DIR / "pending"    # 검증 대기
VALIDATED_DIR = PIPELINE_DATA_DIR / "validated"  # 검증 완료 (학습용)
EVAL_DIR = PIPELINE_DATA_DIR / "eval"          # 평가용 데이터
MODELS_DIR = PIPELINE_DATA_DIR / "models"      # 학습된 모델
LOGS_DIR = PIPELINE_DATA_DIR / "logs"          # 학습/평가 로그
DEPLOY_DIR = PIPELINE_DATA_DIR / "deployed"    # 배포된 모델

for d in [RAW_DIR, PENDING_DIR, VALIDATED_DIR, EVAL_DIR, MODELS_DIR, LOGS_DIR, DEPLOY_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ---------- 1단계: 데이터 수집 ----------

class DataCollector:
    """실시간 통화 데이터를 학습용으로 수집합니다."""

    def __init__(self):
        self.stats = {"collected": 0, "skipped": 0, "errors": 0}

    def save_call(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        segments: List[Dict],
        session_id: str,
        hotwords: List[str] = None,
        metadata: Dict = None,
    ) -> Optional[Path]:
        """통화 데이터를 원본 디렉토리에 저장합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        call_id = f"{timestamp}_{session_id}"
        call_dir = RAW_DIR / call_id
        call_dir.mkdir(exist_ok=True)

        # 오디오 저장
        audio_path = call_dir / f"{call_id}.wav"
        self._save_wav(audio_data, sample_rate, audio_path)

        # 전사 텍스트 추출
        full_text = " ".join(
            seg.get("Content", seg.get("text", "")) for seg in segments
        )

        # 오디오 길이
        duration = len(audio_data) / sample_rate

        # JSON 레이블 생성
        label = {
            "audio_path": f"{call_id}.wav",
            "audio_duration": round(duration, 2),
            "segments": [
                {
                    "speaker": seg.get("Speaker", 0),
                    "text": seg.get("Content", seg.get("text", "")),
                    "start": round(seg.get("Start", 0.0), 2),
                    "end": round(seg.get("End", 0.0), 2),
                }
                for seg in segments
            ],
            "customized_context": hotwords or [],
            "metadata": {
                "session_id": session_id,
                "collected_at": datetime.now().isoformat(),
                "sample_rate": sample_rate,
                "duration": round(duration, 2),
                "status": "raw",
                **(metadata or {}),
            },
        }

        label_path = call_dir / f"{call_id}.json"
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        self.stats["collected"] += 1
        logger.info(f"수집 완료: {call_id} ({duration:.1f}초, {len(segments)}세그먼트)")
        return call_dir

    def import_existing_data(
        self,
        data_dir: str,
        audio_ext: str = ".wav",
        label_ext: str = ".json",
        verified: bool = True,
    ) -> Dict[str, int]:
        """기존 전사 데이터를 파이프라인에 임포트합니다.

        이미 사람이 검수한 전사 데이터(예: 1600시간)가 있을 때 사용합니다.
        verified=True면 검수 완료로 간주하고 바로 학습/평가용으로 분류합니다.

        지원 형식:
          1) {name}.wav + {name}.json (1:1 매칭)
          2) 디렉토리 안에 audio.wav + label.json
          3) JSON 안에 "audio_path" 필드로 오디오 경로 지정

        JSON 최소 형식:
          {"segments": [{"text": "...", "start": 0.0, "end": 1.0}]}
          또는
          {"text": "전체 전사 텍스트"}
        """
        src = Path(data_dir)
        if not src.exists():
            logger.error(f"경로가 존재하지 않습니다: {data_dir}")
            return {"imported": 0, "skipped": 0, "errors": 0}

        stats = {"imported": 0, "skipped": 0, "errors": 0}

        # JSON 파일 목록
        json_files = sorted(src.rglob(f"*{label_ext}"))
        logger.info(f"임포트 대상: {len(json_files)}건 ({data_dir})")

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    label = json.load(f)

                # 오디오 파일 찾기
                audio_path = None
                if "audio_path" in label:
                    candidate = json_path.parent / label["audio_path"]
                    if candidate.exists():
                        audio_path = candidate
                if not audio_path:
                    candidate = json_path.with_suffix(audio_ext)
                    if candidate.exists():
                        audio_path = candidate
                if not audio_path:
                    for wav in json_path.parent.glob(f"*{audio_ext}"):
                        audio_path = wav
                        break

                if not audio_path:
                    stats["skipped"] += 1
                    continue

                # 세그먼트 정규화
                segments = label.get("segments", [])
                if not segments and "text" in label:
                    segments = [{"text": label["text"], "start": 0.0, "end": 0.0}]

                if not segments:
                    stats["skipped"] += 1
                    continue

                # 세그먼트 형식 통일
                normalized = []
                for seg in segments:
                    normalized.append({
                        "speaker": seg.get("speaker", seg.get("Speaker", 0)),
                        "text": seg.get("text", seg.get("Content", "")),
                        "start": round(seg.get("start", seg.get("Start", 0.0)), 2),
                        "end": round(seg.get("end", seg.get("End", 0.0)), 2),
                    })

                # 대상 디렉토리 결정
                call_id = json_path.stem
                if verified:
                    # 검수 완료 → 바로 학습/평가용
                    hash_val = int(hashlib.md5(call_id.encode()).hexdigest(), 16)
                    if (hash_val % 100) < 10:  # 10% 평가
                        dest_dir = EVAL_DIR / call_id
                    else:
                        dest_dir = VALIDATED_DIR / call_id
                else:
                    dest_dir = RAW_DIR / call_id

                if dest_dir.exists():
                    stats["skipped"] += 1
                    continue

                dest_dir.mkdir(parents=True)

                # 오디오 복사
                shutil.copy2(audio_path, dest_dir / audio_path.name)

                # 통일된 레이블 저장
                new_label = {
                    "audio_path": audio_path.name,
                    "audio_duration": label.get("audio_duration", label.get("duration", 0)),
                    "segments": normalized,
                    "customized_context": label.get("customized_context",
                                                     label.get("hotwords", [])),
                    "metadata": {
                        "imported_from": str(json_path),
                        "imported_at": datetime.now().isoformat(),
                        "review_status": "approved" if verified else "raw",
                        "source": "import",
                    },
                }
                with open(dest_dir / f"{call_id}.json", "w", encoding="utf-8") as f:
                    json.dump(new_label, f, ensure_ascii=False, indent=2)

                stats["imported"] += 1
                if stats["imported"] % 100 == 0:
                    logger.info(f"임포트 진행: {stats['imported']}건...")

            except Exception as e:
                stats["errors"] += 1
                logger.warning(f"임포트 오류 [{json_path.name}]: {e}")

        logger.info(
            f"임포트 완료: {stats['imported']}건 성공, "
            f"{stats['skipped']}건 스킵, {stats['errors']}건 오류"
        )
        return stats

    def _save_wav(self, audio: np.ndarray, sample_rate: int, path: Path):
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())


# ---------- 2단계: 데이터 검증 ----------

# 검수 상태
REVIEW_STATUS_PENDING = "pending"     # 검수 대기
REVIEW_STATUS_APPROVED = "approved"   # 사람이 검수 완료
REVIEW_STATUS_REJECTED = "rejected"   # 품질 불량 → 학습 제외
REVIEW_STATUS_AUTO = "auto_approved"  # 자동 승인 (높은 신뢰도)

REVIEW_DIR = PIPELINE_DATA_DIR / "review"  # 사람 검수 대기 큐
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


class DataValidator:
    """수집된 데이터를 검증하고, 사람 검수 후 학습/평가용으로 분류합니다.

    핵심: ASR 초기 WER이 높으므로 자동 전사 결과를 그대로 학습하면
    오류가 강화됩니다. 따라서:

    1) 자동 품질 필터 → 명백히 불량한 데이터 제거
    2) 신뢰도 점수 → 높은 신뢰도(≥0.85)는 자동 승인, 나머지는 사람 검수
    3) 사람 검수 UI → 검수자가 전사 텍스트를 교정 후 승인
    4) 승인된 데이터만 학습에 사용
    """

    MIN_DURATION = 3.0       # 최소 3초
    MAX_DURATION = 600.0     # 최대 10분
    MIN_SEGMENTS = 1         # 최소 1개 세그먼트
    MIN_TEXT_LENGTH = 5      # 세그먼트 최소 글자 수
    EVAL_RATIO = 0.1         # 10%는 평가용
    AUTO_APPROVE_THRESHOLD = 0.85  # 이 신뢰도 이상이면 자동 승인

    def __init__(self):
        self.stats = {
            "valid": 0, "invalid": 0, "eval": 0, "train": 0,
            "auto_approved": 0, "pending_review": 0,
        }

    def validate_all(self, auto_approve: bool = True) -> Dict[str, int]:
        """RAW_DIR의 모든 데이터를 검증합니다.

        Args:
            auto_approve: True면 신뢰도 높은 데이터 자동 승인, False면 전부 검수 대기
        """
        for call_dir in sorted(RAW_DIR.iterdir()):
            if not call_dir.is_dir():
                continue
            self._validate_call(call_dir, auto_approve=auto_approve)

        logger.info(
            f"검증 완료: 유효 {self.stats['valid']} / 무효 {self.stats['invalid']} | "
            f"자동승인 {self.stats['auto_approved']}, 검수대기 {self.stats['pending_review']}, "
            f"학습 {self.stats['train']}, 평가 {self.stats['eval']}"
        )
        return self.stats

    def process_reviewed(self) -> Dict[str, int]:
        """사람이 검수 완료한 데이터를 학습/평가용으로 이동합니다."""
        moved = {"train": 0, "eval": 0, "rejected": 0}

        for call_dir in sorted(REVIEW_DIR.iterdir()):
            if not call_dir.is_dir():
                continue

            json_files = list(call_dir.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r", encoding="utf-8") as f:
                label = json.load(f)

            status = label.get("metadata", {}).get("review_status")

            if status == REVIEW_STATUS_REJECTED:
                moved["rejected"] += 1
                continue

            if status == REVIEW_STATUS_APPROVED:
                # 검수 완료 → 학습/평가 분류
                hash_val = int(hashlib.md5(call_dir.name.encode()).hexdigest(), 16)
                if (hash_val % 100) < (self.EVAL_RATIO * 100):
                    dest = EVAL_DIR / call_dir.name
                    moved["eval"] += 1
                else:
                    dest = VALIDATED_DIR / call_dir.name
                    moved["train"] += 1

                if not dest.exists():
                    shutil.copytree(call_dir, dest)
                    logger.info(f"검수 완료 → {'평가' if 'eval' in str(dest) else '학습'}: {call_dir.name}")

        logger.info(f"검수 처리: 학습 {moved['train']}, 평가 {moved['eval']}, 거부 {moved['rejected']}")
        return moved

    def _validate_call(self, call_dir: Path, auto_approve: bool = True):
        json_files = list(call_dir.glob("*.json"))
        if not json_files:
            self.stats["invalid"] += 1
            return

        label_path = json_files[0]
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        # 이미 처리된 데이터면 스킵
        review_status = label.get("metadata", {}).get("review_status")
        if review_status in (REVIEW_STATUS_APPROVED, REVIEW_STATUS_AUTO, REVIEW_STATUS_REJECTED):
            return

        # 기본 품질 체크
        issues = []

        audio_filename = label.get("audio_path", "")
        audio_path = call_dir / audio_filename
        if not audio_path.exists():
            issues.append("오디오 파일 없음")

        duration = label.get("audio_duration", 0)
        if duration < self.MIN_DURATION:
            issues.append(f"너무 짧음 ({duration:.1f}초)")
        if duration > self.MAX_DURATION:
            issues.append(f"너무 김 ({duration:.1f}초)")

        segments = label.get("segments", [])
        if len(segments) < self.MIN_SEGMENTS:
            issues.append("세그먼트 없음")

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if len(text) < self.MIN_TEXT_LENGTH:
                issues.append(f"세그먼트 {i} 텍스트 너무 짧음")
                break

        for i, seg in enumerate(segments):
            if seg.get("start", 0) > seg.get("end", 0):
                issues.append(f"세그먼트 {i} 시간 역전")
                break

        if issues:
            self.stats["invalid"] += 1
            logger.warning(f"검증 실패 [{call_dir.name}]: {', '.join(issues)}")
            return

        self.stats["valid"] += 1

        # 신뢰도 점수 계산
        confidence = self._compute_confidence(label, segments)
        label.setdefault("metadata", {})["confidence"] = round(confidence, 3)

        if auto_approve and confidence >= self.AUTO_APPROVE_THRESHOLD:
            # 높은 신뢰도 → 자동 승인 → 바로 학습/평가용으로
            label["metadata"]["review_status"] = REVIEW_STATUS_AUTO
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            hash_val = int(hashlib.md5(call_dir.name.encode()).hexdigest(), 16)
            if (hash_val % 100) < (self.EVAL_RATIO * 100):
                dest = EVAL_DIR / call_dir.name
                self.stats["eval"] += 1
            else:
                dest = VALIDATED_DIR / call_dir.name
                self.stats["train"] += 1

            if not dest.exists():
                shutil.copytree(call_dir, dest)

            self.stats["auto_approved"] += 1
            logger.info(f"자동 승인 (신뢰도 {confidence:.2f}): {call_dir.name}")
        else:
            # 낮은 신뢰도 → 사람 검수 대기 큐로
            label["metadata"]["review_status"] = REVIEW_STATUS_PENDING
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            dest = REVIEW_DIR / call_dir.name
            if not dest.exists():
                shutil.copytree(call_dir, dest)

            self.stats["pending_review"] += 1
            logger.info(f"검수 대기 (신뢰도 {confidence:.2f}): {call_dir.name}")

    def _compute_confidence(self, label: Dict, segments: List[Dict]) -> float:
        """ASR 전사 결과의 신뢰도를 추정합니다.

        완벽한 신뢰도는 아니지만, 명백히 나쁜 전사를 걸러내는 데 유용합니다.
        0.0 (매우 불확실) ~ 1.0 (매우 확실)
        """
        score = 0.5  # 기본 점수

        # 1. 세그먼트 수 - 119 통화는 보통 여러 발화가 있음
        seg_count = len(segments)
        if 3 <= seg_count <= 30:
            score += 0.1
        elif seg_count > 30:
            score -= 0.1  # 너무 많은 세그먼트는 잘못된 분할 가능성

        # 2. 텍스트 품질 지표
        full_text = " ".join(seg.get("text", "") for seg in segments)
        text_len = len(full_text)

        # 한국어 비율 체크 (한글이 대부분이어야 함)
        import re
        korean_chars = len(re.findall(r"[가-힣]", full_text))
        if text_len > 0:
            korean_ratio = korean_chars / text_len
            if korean_ratio >= 0.6:
                score += 0.15
            elif korean_ratio < 0.3:
                score -= 0.2  # 한국어가 너무 적으면 오인식 가능성

        # 3. 119 관련 키워드 존재 여부 (도메인 적합성)
        domain_keywords = [
            "119", "신고", "화재", "불", "구급", "구조", "사고",
            "환자", "부상", "의식", "호흡", "출혈",
            "여보세요", "네", "아파트", "도로", "건물",
        ]
        keyword_hits = sum(1 for kw in domain_keywords if kw in full_text)
        if keyword_hits >= 3:
            score += 0.15
        elif keyword_hits >= 1:
            score += 0.05

        # 4. 주소 패턴 존재 (도로명, 지번 등)
        addr_patterns = [
            r"[가-힣]+(?:로|대로|길)\s*\d+",  # 도로명
            r"[가-힣]+[동리읍면]\s*\d+",        # 지번
            r"[가-힣]+[시군구]",                 # 시군구
        ]
        for pat in addr_patterns:
            if re.search(pat, full_text):
                score += 0.05

        # 5. 반복 텍스트 감지 (ASR 루핑 오류)
        if seg_count >= 3:
            texts = [seg.get("text", "") for seg in segments]
            unique_texts = set(texts)
            if len(unique_texts) < len(texts) * 0.5:
                score -= 0.3  # 절반 이상이 동일 텍스트면 ASR 오류

        # 6. 평균 세그먼트 길이 (너무 길거나 짧으면 의심)
        avg_seg_len = text_len / max(seg_count, 1)
        if 5 <= avg_seg_len <= 100:
            score += 0.05
        elif avg_seg_len > 200:
            score -= 0.1  # 하나의 세그먼트에 너무 많은 텍스트

        return max(0.0, min(1.0, score))

    @staticmethod
    def get_review_queue() -> List[Dict]:
        """검수 대기 중인 데이터 목록을 반환합니다."""
        queue = []
        for call_dir in sorted(REVIEW_DIR.iterdir()):
            if not call_dir.is_dir():
                continue
            json_files = list(call_dir.glob("*.json"))
            if not json_files:
                continue
            with open(json_files[0], "r", encoding="utf-8") as f:
                label = json.load(f)
            status = label.get("metadata", {}).get("review_status", "")
            if status == REVIEW_STATUS_PENDING:
                queue.append({
                    "call_id": call_dir.name,
                    "duration": label.get("audio_duration", 0),
                    "segments": label.get("segments", []),
                    "confidence": label.get("metadata", {}).get("confidence", 0),
                    "collected_at": label.get("metadata", {}).get("collected_at", ""),
                })
        return queue

    @staticmethod
    def submit_review(call_id: str, corrected_segments: List[Dict], approved: bool) -> bool:
        """검수 결과를 제출합니다.

        Args:
            call_id: 통화 ID
            corrected_segments: 교정된 세그먼트 (사람이 수정한 텍스트)
            approved: True면 승인 (교정 텍스트로 학습), False면 거부
        """
        call_dir = REVIEW_DIR / call_id
        if not call_dir.exists():
            return False

        json_files = list(call_dir.glob("*.json"))
        if not json_files:
            return False

        label_path = json_files[0]
        with open(label_path, "r", encoding="utf-8") as f:
            label = json.load(f)

        if approved:
            # 교정된 텍스트로 세그먼트 업데이트
            label["segments"] = corrected_segments
            label["metadata"]["review_status"] = REVIEW_STATUS_APPROVED
            label["metadata"]["reviewed_at"] = datetime.now().isoformat()
            # 원본 ASR 결과도 보존 (나중에 WER 비교용)
            label["metadata"]["original_asr_segments"] = label.get("segments", [])
        else:
            label["metadata"]["review_status"] = REVIEW_STATUS_REJECTED
            label["metadata"]["reviewed_at"] = datetime.now().isoformat()

        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        logger.info(f"검수 {'승인' if approved else '거부'}: {call_id}")
        return True


# ---------- 3단계: 학습 ----------

class ContinuousTrainer:
    """지속적으로 LoRA 학습을 수행합니다."""

    def __init__(
        self,
        base_model: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
    ):
        self.base_model = base_model
        self.device = device
        self.current_version = self._get_latest_version()

    def _get_latest_version(self) -> int:
        versions = [
            int(d.name.split("_v")[-1])
            for d in MODELS_DIR.iterdir()
            if d.is_dir() and "_v" in d.name
        ]
        return max(versions) if versions else 0

    def _prepare_flat_dataset(self) -> Path:
        """검증된 데이터를 학습 스크립트가 읽을 수 있는 flat 디렉토리로 복사합니다."""
        flat_dir = PIPELINE_DATA_DIR / "train_flat"
        if flat_dir.exists():
            shutil.rmtree(flat_dir)
        flat_dir.mkdir()

        count = 0
        for call_dir in sorted(VALIDATED_DIR.iterdir()):
            if not call_dir.is_dir():
                continue
            for f in call_dir.iterdir():
                dest = flat_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
            count += 1

        # 기존 119 학습 데이터도 포함
        seed_dir = FINETUNE_DIR / "119_dataset"
        if seed_dir.exists():
            for f in seed_dir.glob("call_*.json"):
                dest = flat_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)

        logger.info(f"학습 데이터 준비: {count}건 + 시드 데이터")
        return flat_dir

    def train(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        lora_r: int = 16,
        resume_from: str = None,
    ) -> Path:
        """LoRA 학습을 실행하고 새 버전 모델을 저장합니다."""
        self.current_version += 1
        version_tag = f"119_asr_v{self.current_version:04d}"
        output_dir = MODELS_DIR / version_tag

        data_dir = self._prepare_flat_dataset()

        # 학습 데이터가 있는지 확인
        json_count = len(list(data_dir.glob("*.json")))
        if json_count == 0:
            logger.error("학습 데이터가 없습니다. 먼저 데이터를 수집/검증하세요.")
            return None

        # base_model 결정: 이전 버전이 있으면 이어서 학습
        model_path = self.base_model
        if resume_from:
            model_path = resume_from
        elif self.current_version > 1:
            prev_version = f"119_asr_v{self.current_version - 1:04d}"
            prev_model = MODELS_DIR / prev_version
            if prev_model.exists():
                model_path = str(prev_model)
                logger.info(f"이전 모델에서 이어서 학습: {prev_version}")

        logger.info(f"학습 시작: {version_tag} (데이터 {json_count}건, epochs={epochs})")

        cmd = [
            sys.executable, str(FINETUNE_DIR / "lora_finetune.py"),
            "--model_path", model_path,
            "--data_dir", str(data_dir),
            "--use_customized_context", "True",
            "--output_dir", str(output_dir),
            "--num_train_epochs", str(epochs),
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", str(learning_rate),
            "--warmup_steps", "50",
            "--logging_steps", "5",
            "--save_steps", "50",
            "--save_total_limit", "3",
            "--bf16", "True" if self.device == "cuda" else "False",
            "--seed", "42",
            "--report_to", "none",
            "--lora_r", str(lora_r),
        ]

        log_path = LOGS_DIR / f"train_{version_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        with open(log_path, "w") as log_file:
            process = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
            )

        if process.returncode != 0:
            logger.error(f"학습 실패 (exit code {process.returncode}). 로그: {log_path}")
            return None

        # 학습 메타데이터 저장
        meta = {
            "version": version_tag,
            "base_model": model_path,
            "data_count": json_count,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "trained_at": datetime.now().isoformat(),
            "log_path": str(log_path),
        }
        with open(output_dir / "pipeline_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"학습 완료: {version_tag} → {output_dir}")
        return output_dir


# ---------- 4단계: 평가 ----------

class WEREvaluator:
    """WER(단어 오류율)을 측정합니다."""

    def __init__(self):
        self.history: List[Dict] = []
        self._load_history()

    def _load_history(self):
        history_path = LOGS_DIR / "wer_history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.history = json.load(f)

    def _save_history(self):
        with open(LOGS_DIR / "wer_history.json", "w") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """WER을 계산합니다 (편집 거리 기반)."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # 편집 거리 (삽입, 삭제, 대치)
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

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """CER(문자 오류율)을 계산합니다 - 한국어에 더 적합."""
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))

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

    def evaluate_model(
        self,
        model_path: str,
        eval_data_dir: str = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """모델의 WER/CER을 평가합니다."""
        eval_dir = Path(eval_data_dir) if eval_data_dir else EVAL_DIR

        # 평가 데이터 수집
        eval_samples = []
        for json_path in sorted(eval_dir.rglob("*.json")):
            if json_path.name.startswith("call_") or json_path.name.startswith("20"):
                with open(json_path) as f:
                    data = json.load(f)
                audio_path = json_path.parent / data.get("audio_path", "")
                if audio_path.exists():
                    eval_samples.append({
                        "audio_path": str(audio_path),
                        "segments": data["segments"],
                        "hotwords": data.get("customized_context", []),
                    })

        if not eval_samples:
            logger.warning("평가 데이터가 없습니다.")
            return {"wer": -1, "cer": -1, "count": 0}

        # ASR 서비스 로드
        from demo.realtime_asr.app import RealtimeASRService
        service = RealtimeASRService(model_path=model_path, device=device)
        service.load()

        total_wer = 0.0
        total_cer = 0.0
        total_addr_wer = 0.0
        addr_count = 0
        count = 0

        for sample in eval_samples:
            if sample["hotwords"]:
                service.set_hotwords(sample["hotwords"])

            try:
                result = service.transcribe(sample["audio_path"])
            except Exception as e:
                logger.warning(f"평가 오류: {e}")
                continue

            # 참조 텍스트
            ref_text = " ".join(seg["text"] for seg in sample["segments"])

            # 추론 텍스트
            hyp_segments = result.get("segments", [])
            hyp_text = " ".join(
                seg.get("Content", seg.get("text", "")) for seg in hyp_segments
            )

            wer = self.compute_wer(ref_text, hyp_text)
            cer = self.compute_cer(ref_text, hyp_text)
            total_wer += wer
            total_cer += cer
            count += 1

            # 주소 부분 WER 별도 측정
            addr_ref = self._extract_address_text(ref_text)
            addr_hyp = self._extract_address_text(hyp_text)
            if addr_ref:
                addr_wer = self.compute_wer(addr_ref, addr_hyp)
                total_addr_wer += addr_wer
                addr_count += 1

        avg_wer = (total_wer / count * 100) if count > 0 else -1
        avg_cer = (total_cer / count * 100) if count > 0 else -1
        avg_addr_wer = (total_addr_wer / addr_count * 100) if addr_count > 0 else -1

        result = {
            "model_path": model_path,
            "wer": round(avg_wer, 2),
            "cer": round(avg_cer, 2),
            "address_wer": round(avg_addr_wer, 2),
            "eval_count": count,
            "evaluated_at": datetime.now().isoformat(),
        }

        self.history.append(result)
        self._save_history()

        logger.info(
            f"평가 결과: WER={avg_wer:.2f}% CER={avg_cer:.2f}% "
            f"주소WER={avg_addr_wer:.2f}% ({count}건)"
        )
        return result

    @staticmethod
    def _extract_address_text(text: str) -> str:
        """텍스트에서 주소 관련 부분을 추출합니다."""
        import re
        patterns = [
            r"[가-힣]+(?:특별시|광역시|도)\s*[가-힣]+[시군구]\s*[가-힣0-9]+(?:로|대로|길|번길)\s*\d+",
            r"[가-힣]+[시군구]\s*[가-힣]+[동리읍면]\s*\d+",
        ]
        matches = []
        for p in patterns:
            matches.extend(re.findall(p, text))
        return " ".join(matches)

    def print_history(self):
        """WER 변화 이력을 출력합니다."""
        if not self.history:
            logger.info("평가 이력이 없습니다.")
            return

        print("\n" + "=" * 70)
        print("  WER 변화 추적")
        print("=" * 70)
        print(f"  {'#':>3}  {'날짜':>19}  {'WER':>7}  {'CER':>7}  {'주소WER':>8}  {'모델'}")
        print("-" * 70)

        best_wer = float("inf")
        for i, h in enumerate(self.history):
            wer = h.get("wer", -1)
            cer = h.get("cer", -1)
            addr_wer = h.get("address_wer", -1)
            date = h.get("evaluated_at", "")[:19]
            model = Path(h.get("model_path", "")).name

            indicator = ""
            if wer > 0 and wer < best_wer:
                best_wer = wer
                indicator = " << BEST"

            print(f"  {i+1:>3}  {date}  {wer:>6.2f}%  {cer:>6.2f}%  {addr_wer:>7.2f}%  {model}{indicator}")

        print("=" * 70)
        if best_wer < float("inf"):
            print(f"  최고 WER: {best_wer:.2f}%")
        print()


# ---------- 5단계: 배포 ----------

class ModelDeployer:
    """학습된 모델을 서빙 서버에 배포합니다."""

    def __init__(self, serving_url: str = "http://localhost:8000"):
        self.serving_url = serving_url

    def deploy(
        self,
        model_dir: Path,
        evaluator: WEREvaluator = None,
        max_wer: float = None,
    ) -> bool:
        """모델을 배포합니다."""
        if not model_dir or not model_dir.exists():
            logger.error(f"모델 디렉토리가 없습니다: {model_dir}")
            return False

        # WER 체크 (배포 게이트)
        if max_wer is not None and evaluator and evaluator.history:
            latest_wer = evaluator.history[-1].get("wer", 100)
            if latest_wer > max_wer:
                logger.warning(
                    f"WER {latest_wer:.2f}%가 기준치 {max_wer:.2f}%를 초과하여 배포 중단"
                )
                return False

        # 배포 디렉토리에 복사
        deploy_name = f"deployed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        deploy_path = DEPLOY_DIR / deploy_name

        # 심볼릭 링크로 current 유지
        current_link = DEPLOY_DIR / "current"

        shutil.copytree(model_dir, deploy_path)

        if current_link.is_symlink() or current_link.exists():
            current_link.unlink()
        current_link.symlink_to(deploy_path)

        # 배포 메타데이터
        meta = {
            "source_model": str(model_dir),
            "deployed_at": datetime.now().isoformat(),
            "deploy_path": str(deploy_path),
        }
        if evaluator and evaluator.history:
            meta["wer_at_deploy"] = evaluator.history[-1].get("wer")
            meta["cer_at_deploy"] = evaluator.history[-1].get("cer")

        with open(deploy_path / "deploy_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"배포 완료: {deploy_path}")
        logger.info(f"현재 모델: {current_link} → {deploy_path}")

        return True


# ---------- 전체 파이프라인 ----------

class Pipeline:
    """수집 → 검증 → 학습 → 평가 → 배포 전체 파이프라인"""

    def __init__(
        self,
        base_model: str = "microsoft/VibeVoice-ASR",
        device: str = "cuda",
        serving_url: str = "http://localhost:8000",
        deploy_wer_threshold: float = 15.0,
    ):
        self.collector = DataCollector()
        self.validator = DataValidator()
        self.trainer = ContinuousTrainer(base_model=base_model, device=device)
        self.evaluator = WEREvaluator()
        self.deployer = ModelDeployer(serving_url=serving_url)
        self.deploy_wer_threshold = deploy_wer_threshold

    def run_cycle(
        self,
        epochs: int = 5,
        learning_rate: float = 1e-4,
    ) -> Dict[str, Any]:
        """학습 사이클 1회를 실행합니다."""
        cycle_start = time.time()
        result = {"status": "started", "steps": {}}

        # 1. 검증
        logger.info("=" * 50)
        logger.info("STEP 1: 데이터 검증")
        logger.info("=" * 50)
        val_stats = self.validator.validate_all()
        result["steps"]["validate"] = val_stats

        # 2. 학습
        logger.info("=" * 50)
        logger.info("STEP 2: LoRA 학습")
        logger.info("=" * 50)
        model_dir = self.trainer.train(epochs=epochs, learning_rate=learning_rate)
        result["steps"]["train"] = {
            "model_dir": str(model_dir) if model_dir else None,
            "version": self.trainer.current_version,
        }

        if not model_dir:
            result["status"] = "train_failed"
            return result

        # 3. 평가
        logger.info("=" * 50)
        logger.info("STEP 3: WER 평가")
        logger.info("=" * 50)
        eval_result = self.evaluator.evaluate_model(str(model_dir))
        result["steps"]["evaluate"] = eval_result

        # 4. 배포 (WER 기준 충족 시)
        logger.info("=" * 50)
        logger.info("STEP 4: 배포 판단")
        logger.info("=" * 50)
        deployed = self.deployer.deploy(
            model_dir,
            evaluator=self.evaluator,
            max_wer=self.deploy_wer_threshold,
        )
        result["steps"]["deploy"] = {"deployed": deployed}

        elapsed = time.time() - cycle_start
        result["status"] = "completed"
        result["elapsed_seconds"] = round(elapsed, 1)

        # 이력 출력
        self.evaluator.print_history()

        logger.info(f"사이클 완료: {elapsed:.0f}초")
        return result

    def run_continuous(
        self,
        interval_hours: float = 6.0,
        epochs: int = 5,
        min_new_samples: int = 10,
    ):
        """지속적으로 학습 사이클을 반복합니다."""
        logger.info(f"지속 학습 시작 (간격: {interval_hours}시간, 최소 신규 데이터: {min_new_samples}건)")

        cycle = 0
        while True:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"  학습 사이클 #{cycle}")
            logger.info(f"{'='*60}")

            # 신규 데이터 확인
            raw_count = sum(1 for d in RAW_DIR.iterdir() if d.is_dir())
            validated_count = sum(1 for d in VALIDATED_DIR.iterdir() if d.is_dir())

            if raw_count < min_new_samples:
                logger.info(
                    f"신규 데이터 부족 ({raw_count}/{min_new_samples}). "
                    f"{interval_hours}시간 후 재시도."
                )
            else:
                result = self.run_cycle(epochs=epochs)

                # 결과 로그 저장
                log_path = LOGS_DIR / f"cycle_{cycle:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

            # 대기
            logger.info(f"다음 사이클까지 {interval_hours}시간 대기...")
            time.sleep(interval_hours * 3600)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="119 ASR 지속 학습 파이프라인")
    parser.add_argument(
        "command",
        choices=["collect", "import", "validate", "review", "train", "evaluate", "deploy", "run", "history", "status"],
        help="실행할 명령",
    )
    parser.add_argument("--model", default="microsoft/VibeVoice-ASR", help="기본 모델 경로")
    parser.add_argument("--device", default="cuda", help="학습/추론 디바이스")
    parser.add_argument("--epochs", type=int, default=5, help="학습 에포크")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--interval", type=float, default=6.0, help="지속 학습 간격 (시간)")
    parser.add_argument("--min-samples", type=int, default=10, help="최소 신규 샘플 수")
    parser.add_argument("--max-wer", type=float, default=15.0, help="배포 WER 기준치 (%)")
    parser.add_argument("--model-dir", help="평가/배포할 모델 디렉토리")
    parser.add_argument("--no-auto-approve", action="store_true", help="자동 승인 비활성화 (전부 사람 검수)")
    parser.add_argument("--data-dir", help="임포트할 기존 전사 데이터 디렉토리")
    parser.add_argument("--unverified", action="store_true", help="임포트 데이터를 미검수로 처리")

    args = parser.parse_args()

    if args.command == "import":
        if not args.data_dir:
            logger.error("--data-dir를 지정하세요 (기존 전사 데이터 경로)")
            logger.info("예: python pipeline.py import --data-dir /path/to/1600h_data")
            logger.info("    python pipeline.py import --data-dir /data/transcribed --unverified")
            return
        collector = DataCollector()
        collector.import_existing_data(
            args.data_dir,
            verified=not args.unverified,
        )

    elif args.command == "validate":
        validator = DataValidator()
        validator.validate_all(auto_approve=not args.no_auto_approve)

    elif args.command == "review":
        validator = DataValidator()
        # 검수 완료된 데이터 처리
        validator.process_reviewed()
        # 대기 목록 출력
        queue = validator.get_review_queue()
        print(f"\n검수 대기: {len(queue)}건")
        for item in queue[:20]:
            print(f"  {item['call_id']} ({item['duration']:.1f}초, 신뢰도 {item['confidence']:.2f})")
        if len(queue) > 20:
            print(f"  ... 외 {len(queue)-20}건")
        print(f"\n검수 UI: http://localhost:8080/review")

    elif args.command == "train":
        trainer = ContinuousTrainer(base_model=args.model, device=args.device)
        trainer.train(epochs=args.epochs, learning_rate=args.lr)

    elif args.command == "evaluate":
        evaluator = WEREvaluator()
        model = args.model_dir or args.model
        evaluator.evaluate_model(model, device=args.device)
        evaluator.print_history()

    elif args.command == "deploy":
        if not args.model_dir:
            logger.error("--model-dir를 지정하세요")
            return
        deployer = ModelDeployer()
        evaluator = WEREvaluator()
        deployer.deploy(Path(args.model_dir), evaluator=evaluator, max_wer=args.max_wer)

    elif args.command == "run":
        pipeline = Pipeline(
            base_model=args.model,
            device=args.device,
            deploy_wer_threshold=args.max_wer,
        )
        pipeline.run_continuous(
            interval_hours=args.interval,
            epochs=args.epochs,
            min_new_samples=args.min_samples,
        )

    elif args.command == "history":
        evaluator = WEREvaluator()
        evaluator.print_history()

    elif args.command == "status":
        print(f"\n{'='*50}")
        print("  119 ASR 파이프라인 현황")
        print(f"{'='*50}")
        print(f"  원본 데이터:   {sum(1 for d in RAW_DIR.iterdir() if d.is_dir())}건")
        print(f"  검증 대기:     {sum(1 for d in PENDING_DIR.iterdir() if d.is_dir())}건")
        print(f"  학습 데이터:   {sum(1 for d in VALIDATED_DIR.iterdir() if d.is_dir())}건")
        print(f"  평가 데이터:   {sum(1 for d in EVAL_DIR.iterdir() if d.is_dir())}건")
        print(f"  학습 모델:     {sum(1 for d in MODELS_DIR.iterdir() if d.is_dir())}개")
        current = DEPLOY_DIR / "current"
        if current.is_symlink():
            print(f"  배포 모델:     {current.resolve().name}")
        else:
            print(f"  배포 모델:     없음")
        print(f"{'='*50}\n")

    elif args.command == "collect":
        logger.info("데이터 수집은 실시간 ASR 서버가 자동으로 수행합니다.")
        logger.info("서버 실행: MODEL_PATH=... python demo/realtime_asr/app.py")
        logger.info(f"수집 경로: {RAW_DIR}")


if __name__ == "__main__":
    main()
