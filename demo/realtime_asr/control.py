"""
119 ASR 통제 모듈

통화 세션 관리, 긴급도 자동 분류, 출동 지령서 생성,
실시간 모니터링 기능을 제공합니다.
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------- 긴급도 분류 ----------

class UrgencyLevel(IntEnum):
    """긴급도 등급 (1 = 최고)"""
    CRITICAL = 1   # 즉시 출동: 의식 없음, 심정지, 대형 화재, 건물 붕괴
    URGENT = 2     # 긴급 출동: 출혈, 호흡곤란, 화재, 교통사고 부상
    NORMAL = 3     # 일반 출동: 경미한 부상, 소규모 화재, 일반 구조


CRITICAL_KEYWORDS = [
    "의식 없", "의식이 없", "숨을 안 쉬", "숨 안 쉬", "심장이 멈", "심정지",
    "심폐소생술", "CPR", "AED",
    "불이 번지", "폭발", "건물 붕괴", "붕괴", "매몰",
    "물에 빠", "익수", "추락",
    "피가 많이", "대량 출혈", "목을 다", "머리를 다",
    "아이가", "아기가", "임산부", "산모",
]

URGENT_KEYWORDS = [
    "출혈", "피가 나", "골절", "부러", "화상",
    "호흡곤란", "숨이 잘", "가슴이 아",
    "쓰러", "넘어", "의식이 흐릿",
    "불이 났", "연기", "가스 냄새", "가스 누출",
    "교통사고", "추돌", "전복", "끼임", "갇힘",
    "말을 못하", "한쪽 팔", "뇌졸중",
    "경련", "발작",
]

NORMAL_KEYWORDS = [
    "좀 다쳤", "살짝", "가벼운", "긁힘",
    "냄새가 나", "이상한 소리",
    "문이 안 열", "잠겨", "갇혔는데 위험",
    "고양이", "강아지", "동물",
]


def classify_urgency(text: str) -> Dict[str, Any]:
    """
    전사 텍스트에서 긴급도를 자동 분류합니다.

    Returns:
        {"level": 1~3, "label": "...", "matched_keywords": [...], "reason": "..."}
    """
    text_lower = text.lower()
    matched_critical = [kw for kw in CRITICAL_KEYWORDS if kw in text_lower]
    matched_urgent = [kw for kw in URGENT_KEYWORDS if kw in text_lower]
    matched_normal = [kw for kw in NORMAL_KEYWORDS if kw in text_lower]

    if matched_critical:
        return {
            "level": UrgencyLevel.CRITICAL,
            "label": "긴급",
            "color": "red",
            "matched_keywords": matched_critical,
            "reason": f"긴급 키워드 감지: {', '.join(matched_critical[:3])}",
        }
    elif matched_urgent:
        return {
            "level": UrgencyLevel.URGENT,
            "label": "준긴급",
            "color": "orange",
            "matched_keywords": matched_urgent,
            "reason": f"준긴급 키워드 감지: {', '.join(matched_urgent[:3])}",
        }
    else:
        return {
            "level": UrgencyLevel.NORMAL,
            "label": "일반",
            "color": "yellow",
            "matched_keywords": matched_normal,
            "reason": "일반 신고" if not matched_normal else f"일반 키워드: {', '.join(matched_normal[:3])}",
        }


# ---------- 신고 유형 분류 ----------

REPORT_TYPE_PATTERNS = {
    "화재": ["화재", "불이 났", "불이 나", "연기", "폭발", "가스 누출", "가스 냄새", "방화", "전기 합선"],
    "구급": ["구급", "쓰러", "의식", "호흡", "심장", "출혈", "골절", "화상", "뇌졸중",
             "경련", "발작", "중독", "알레르기", "심폐소생", "다쳤", "아파"],
    "구조": ["구조", "갇힘", "끼임", "고립", "매몰", "붕괴", "물에 빠", "추락", "고양이", "강아지"],
    "교통사고": ["교통사고", "추돌", "전복", "중앙선", "역주행", "오토바이", "보행자", "차가 부딪", "차가 넘어"],
}


def classify_report_type(text: str) -> Dict[str, Any]:
    """신고 유형을 분류합니다."""
    scores = {}
    for rtype, keywords in REPORT_TYPE_PATTERNS.items():
        matched = [kw for kw in keywords if kw in text]
        scores[rtype] = len(matched)

    if not any(scores.values()):
        return {"type": "기타", "confidence": 0.0, "matched": []}

    primary = max(scores, key=scores.get)
    total_matched = sum(scores.values())
    confidence = scores[primary] / total_matched if total_matched > 0 else 0.0

    return {
        "type": primary,
        "confidence": round(confidence, 2),
        "matched": [kw for kw in REPORT_TYPE_PATTERNS[primary] if kw in text],
        "all_scores": scores,
    }


# ---------- 환자/피해자 정보 추출 ----------

AGE_PATTERN = re.compile(r"(?P<age>\d{1,3})\s*(?:세|살)")
GENDER_PATTERN = re.compile(r"(?P<gender>남성|여성|남자|여자|할아버지|할머니|아이|아기|아동|어린이|산모|임산부)")
COUNT_PATTERN = re.compile(r"(?P<count>\d+)\s*(?:명|분|사람)")


def extract_patient_info(text: str) -> List[Dict[str, Any]]:
    """전사 텍스트에서 환자/피해자 정보를 추출합니다."""
    patients = []

    ages = AGE_PATTERN.findall(text)
    genders = GENDER_PATTERN.findall(text)
    counts = COUNT_PATTERN.findall(text)

    # 성별 정규화
    gender_map = {
        "남성": "남성", "남자": "남성", "할아버지": "남성(노인)",
        "여성": "여성", "여자": "여성", "할머니": "여성(노인)",
        "아이": "아동", "아기": "영아", "아동": "아동", "어린이": "아동",
        "산모": "여성(산모)", "임산부": "여성(산모)",
    }

    # 증상 추출
    symptom_keywords = [
        "의식 없", "의식이 없", "호흡곤란", "숨이", "출혈", "피가 나",
        "골절", "부러", "화상", "경련", "발작", "구토",
        "가슴 통증", "가슴이 아", "두통", "어지러",
        "말을 못", "한쪽 팔", "한쪽 다리",
        "움직이지 못", "못 움직",
    ]
    symptoms = [kw for kw in symptom_keywords if kw in text]

    # 피해자 수
    total_count = int(counts[0]) if counts else 1

    for i in range(min(total_count, len(ages) if ages else 1)):
        patient = {
            "index": i + 1,
            "age": f"{ages[i]}세" if i < len(ages) else "불명",
            "gender": gender_map.get(genders[i], genders[i]) if i < len(genders) else "불명",
            "symptoms": symptoms,
        }
        patients.append(patient)

    if not patients:
        patients.append({
            "index": 1,
            "age": f"{ages[0]}세" if ages else "불명",
            "gender": gender_map.get(genders[0], genders[0]) if genders else "불명",
            "symptoms": symptoms,
        })

    return patients


# ---------- 통화 세션 관리 ----------

@dataclass
class CallSession:
    """119 통화 세션"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    call_sid: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    status: str = "active"  # active, processing, completed, dispatched
    transcripts: List[Dict[str, Any]] = field(default_factory=list)
    locations: List[Dict[str, Any]] = field(default_factory=list)
    urgency: Optional[Dict[str, Any]] = None
    report_type: Optional[Dict[str, Any]] = None
    patients: List[Dict[str, Any]] = field(default_factory=list)
    dispatch_order: Optional[Dict[str, Any]] = None
    operator_notes: str = ""
    audio_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "call_sid": self.call_sid,
            "started_at": self.started_at,
            "status": self.status,
            "transcript_count": len(self.transcripts),
            "locations": self.locations,
            "urgency": self.urgency,
            "report_type": self.report_type,
            "patients": self.patients,
            "dispatch_order": self.dispatch_order,
            "operator_notes": self.operator_notes,
            "audio_duration": self.audio_duration,
        }


class SessionManager:
    """통화 세션 관리자"""

    def __init__(self):
        self.sessions: Dict[str, CallSession] = {}
        self._stats = {
            "total_calls": 0,
            "active_calls": 0,
            "dispatched": 0,
            "avg_response_time": 0.0,
        }

    def create_session(self, call_sid: Optional[str] = None) -> CallSession:
        session = CallSession(call_sid=call_sid)
        self.sessions[session.session_id] = session
        self._stats["total_calls"] += 1
        self._stats["active_calls"] += 1
        logger.info(f"세션 생성: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[CallSession]:
        return self.sessions.get(session_id)

    def update_transcript(self, session_id: str, result: Dict[str, Any]):
        """ASR 결과를 세션에 추가하고 자동 분석합니다."""
        session = self.sessions.get(session_id)
        if not session:
            return

        session.transcripts.append(result)

        # 위치 정보 갱신
        if "locations" in result:
            session.locations = result["locations"]

        # 전체 텍스트로 분석
        full_text = self._get_full_text(session)

        # 긴급도 분류
        session.urgency = classify_urgency(full_text)

        # 신고 유형 분류
        session.report_type = classify_report_type(full_text)

        # 환자 정보 추출
        session.patients = extract_patient_info(full_text)

    def _get_full_text(self, session: CallSession) -> str:
        texts = []
        for t in session.transcripts:
            for seg in t.get("segments", []):
                texts.append(seg.get("Content", seg.get("text", "")))
        return " ".join(texts)

    def generate_dispatch_order(self, session_id: str) -> Optional[Dict[str, Any]]:
        """출동 지령서를 생성합니다."""
        session = self.sessions.get(session_id)
        if not session:
            return None

        full_text = self._get_full_text(session)
        urgency = session.urgency or classify_urgency(full_text)
        report_type = session.report_type or classify_report_type(full_text)
        patients = session.patients or extract_patient_info(full_text)

        # 주소 정보 정리
        location_info = {}
        if session.locations:
            loc = session.locations[0]
            geo = loc.get("geocode")
            location_info = {
                "address": loc.get("extracted", {}).get("full", "주소 미확인"),
                "latitude": geo.get("latitude") if geo else None,
                "longitude": geo.get("longitude") if geo else None,
                "map_urls": loc.get("map_urls", {}),
            }

        # 출동 차량 결정
        vehicles = []
        rtype = report_type.get("type", "기타")
        if rtype == "화재":
            vehicles = ["펌프차", "물탱크차"]
            if urgency["level"] == UrgencyLevel.CRITICAL:
                vehicles += ["고가사다리차", "구급차"]
        elif rtype == "구급":
            vehicles = ["구급차"]
            if urgency["level"] == UrgencyLevel.CRITICAL:
                vehicles.append("닥터카")
        elif rtype == "구조":
            vehicles = ["구조공작차"]
            if "물에 빠" in full_text or "익수" in full_text:
                vehicles.append("수난구조차")
        elif rtype == "교통사고":
            vehicles = ["구급차", "펌프차"]
            count = sum(1 for p in patients if p.get("symptoms"))
            if count > 2:
                vehicles.append("구급차(추가)")

        dispatch = {
            "dispatch_id": f"D-{session.session_id}-{int(time.time()) % 10000:04d}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session.session_id,
            "urgency": {
                "level": urgency["level"],
                "label": urgency["label"],
            },
            "report_type": rtype,
            "location": location_info,
            "patients": patients,
            "patient_count": len(patients),
            "vehicles": vehicles,
            "instructions": self._generate_instructions(rtype, urgency, full_text),
            "full_transcript": full_text[:500],
        }

        session.dispatch_order = dispatch
        session.status = "dispatched"
        self._stats["dispatched"] += 1
        self._stats["active_calls"] -= 1

        logger.info(
            f"출동 지령 생성: {dispatch['dispatch_id']} "
            f"[{urgency['label']}] {rtype} → {vehicles}"
        )
        return dispatch

    def _generate_instructions(
        self, report_type: str, urgency: Dict, text: str
    ) -> List[str]:
        """출동 대원 지시사항을 생성합니다."""
        instructions = []

        if report_type == "화재":
            instructions.append("방화복 착용, 공기호흡기 준비")
            if "가스" in text:
                instructions.append("가스 누출 주의 - 전기 차단 필요")
            if "지하" in text:
                instructions.append("지하 공간 - 배연 장비 준비")
            if "고층" in text or "아파트" in text:
                instructions.append("고층 건물 - 사다리차 배치 고려")

        elif report_type == "구급":
            if "의식 없" in text or "심정지" in text:
                instructions.append("심폐소생술 장비 최우선 준비")
                instructions.append("AED 즉시 사용 준비")
            if "뇌졸중" in text or "말을 못" in text:
                instructions.append("뇌졸중 의심 - 골든타임 확보, 뇌졸중 센터 연락")
            if "골절" in text:
                instructions.append("부목/고정 장비 준비")
            if "출혈" in text or "피가" in text:
                instructions.append("지혈 장비 및 수혈용 혈액 확인")

        elif report_type == "구조":
            if "물" in text or "익수" in text:
                instructions.append("수난구조 장비 - 구명조끼, 구명환 준비")
            if "갇힘" in text or "끼임" in text:
                instructions.append("유압 절단기 및 확장기 준비")
            if "붕괴" in text or "매몰" in text:
                instructions.append("탐색 장비 및 중장비 요청")

        elif report_type == "교통사고":
            instructions.append("교통 통제 및 2차 사고 방지")
            if "전복" in text:
                instructions.append("차량 전복 - 안정화 장비 준비")
            if "끼임" in text or "못 나오" in text:
                instructions.append("인명 구출 장비 - 유압 절단기 준비")

        if urgency["level"] == UrgencyLevel.CRITICAL:
            instructions.insert(0, "*** 최우선 출동 ***")

        return instructions

    def close_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if session and session.status == "active":
            session.status = "completed"
            self._stats["active_calls"] = max(0, self._stats["active_calls"] - 1)

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        return [
            s.to_dict() for s in self.sessions.values()
            if s.status in ("active", "processing")
        ]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.sessions.values()]

    def get_stats(self) -> Dict[str, Any]:
        active = [s for s in self.sessions.values() if s.status == "active"]
        urgency_counts = {"긴급": 0, "준긴급": 0, "일반": 0}
        for s in active:
            if s.urgency:
                label = s.urgency.get("label", "일반")
                urgency_counts[label] = urgency_counts.get(label, 0) + 1

        return {
            "total_calls": self._stats["total_calls"],
            "active_calls": len(active),
            "dispatched": self._stats["dispatched"],
            "completed": sum(1 for s in self.sessions.values() if s.status == "completed"),
            "urgency_distribution": urgency_counts,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
