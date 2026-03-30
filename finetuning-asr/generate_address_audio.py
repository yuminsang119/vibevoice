#!/usr/bin/env python
"""
119 신고전화 주소 음성 합성 데이터 생성기

전국 주소 데이터를 기반으로 TTS 음성을 합성하고,
전화 환경 노이즈를 추가하여 학습 데이터를 대량 생성합니다.

사용법:
    # edge-tts 사용 (무료, 한국어 지원)
    pip install edge-tts numpy
    python generate_address_audio.py --output-dir ./address_data --count 500

    # 증강 비활성화 (깨끗한 음성만)
    python generate_address_audio.py --output-dir ./address_data --count 500 --no-augment

    # 생성 후 파이프라인 임포트
    python demo/realtime_asr/pipeline.py import --data-dir ./address_data
"""

import argparse
import asyncio
import json
import logging
import os
import random
import struct
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("address_audio_gen")

SCRIPT_DIR = Path(__file__).parent
HOTWORDS_PATH = SCRIPT_DIR / "119_dataset" / "nationwide_address_hotwords.json"


# ---------- 주소 생성기 ----------

class AddressGenerator:
    """전국 주소 데이터에서 랜덤 주소를 생성합니다."""

    def __init__(self, hotwords_path: str = None):
        path = Path(hotwords_path) if hotwords_path else HOTWORDS_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.regions = self.data.get("전국_시도", {})
        self.highways = self.data.get("고속도로_IC_JC", {})
        self.common = self.data.get("공통_주소_패턴_키워드", {})

    def random_road_address(self) -> Tuple[str, List[str]]:
        """랜덤 도로명주소 생성"""
        region_name = random.choice(list(self.regions.keys()))
        region = self.regions[region_name]

        # 구/군
        districts = region.get("구", region.get("구군", region.get("주요시", [])))
        district = random.choice(districts) if districts else ""

        # 도로
        roads = region.get("주요도로", [])
        road = random.choice(roads) if roads else "중앙로"

        # 번호
        number = random.randint(1, 1500)

        # 건물명 (선택)
        building = ""
        if random.random() < 0.4:
            landmarks = region.get("랜드마크", [])
            if landmarks:
                building = random.choice(landmarks)

        if building:
            addr = f"{region_name} {district} {road} {number} {building}"
        else:
            addr = f"{region_name} {district} {road} {number}"

        hotwords = [w for w in [district, road, building] if w]
        return addr, hotwords

    def random_jibeon_address(self) -> Tuple[str, List[str]]:
        """랜덤 지번주소 생성"""
        region_name = random.choice(list(self.regions.keys()))
        region = self.regions[region_name]

        districts = region.get("구", region.get("구군", region.get("주요시", [])))
        district = random.choice(districts) if districts else ""

        dong_suffixes = ["동", "리", "읍", "면"]
        dong = random.choice(["삼성", "역삼", "서초", "반포", "대치", "잠실", "논현", "청담",
                               "신당", "장안", "면목", "상봉", "중화", "망우", "둔산", "월평",
                               "노은", "전민", "관평"]) + random.choice(dong_suffixes)

        number = random.randint(1, 999)
        sub = f"-{random.randint(1, 30)}" if random.random() < 0.4 else ""

        addr = f"{region_name} {district} {dong} {number}{sub}번지"
        hotwords = [w for w in [district, dong] if w]
        return addr, hotwords

    def random_intersection(self) -> Tuple[str, List[str]]:
        """랜덤 교차로 주소 생성"""
        region_name = random.choice(list(self.regions.keys()))
        region = self.regions[region_name]

        crosses = region.get("주요교차로", [])
        if crosses:
            cross = random.choice(crosses)
        else:
            roads = region.get("주요도로", ["중앙로"])
            cross = random.choice(roads).rstrip("로대길") + "사거리"

        refs = ["CU편의점 앞", "스타벅스 옆", "맥도날드 건너편", "버스정류장 앞",
                "GS25 맞은편", "우리은행 앞", "지하철 출구 근처", "횡단보도 위",
                "코너 쪽", "모퉁이", "신호등 앞"]
        ref = random.choice(refs)

        addr = f"{cross} {ref}"
        hotwords = [cross]
        return addr, hotwords

    def random_landmark(self) -> Tuple[str, List[str]]:
        """랜덤 랜드마크 기반 주소 생성"""
        region_name = random.choice(list(self.regions.keys()))
        region = self.regions[region_name]

        landmarks = region.get("랜드마크", [])
        if not landmarks:
            return self.random_road_address()

        landmark = random.choice(landmarks)

        details = [
            "정문 앞", "후문 쪽", "주차장", "1층", "2층", "3층", "지하1층",
            "옥상", "로비", "입구", "동쪽 출구", "서쪽 입구", "옆 골목",
            "건너편", "뒤편 주차장", "앞 도로", "맞은편"
        ]
        detail = random.choice(details)

        addr = f"{landmark} {detail}"
        hotwords = [landmark]
        return addr, hotwords

    def random_colloquial(self) -> Tuple[str, List[str]]:
        """랜덤 구어체 위치 설명 생성"""
        templates = [
            "{landmark} 지나서 큰 사거리에서 {direction}으로 꺾으면 {ref} 있는데 거기 {detail}",
            "지하철 {station}역 {exit}번 출구 나와서 쭉 직진하면 {ref} 보이는데 그 {rel}",
            "{landmark} {rel}에 {ref} 있잖아요 거기서 {distance}쯤 {direction}으로 가면",
            "여기 {ref} 앞인데 큰 도로에서 골목으로 들어와서 {distance} 정도 오면",
            "{road} 따라가다 보면 {ref} 나오는데 거기 {detail}",
        ]

        landmarks = ["이마트", "홈플러스", "코스트코", "맥도날드", "스타벅스",
                     "교회", "성당", "학교", "우체국", "파출소", "소방서"]
        stations = ["강남", "홍대입구", "잠실", "신촌", "건대입구", "여의도",
                    "사당", "교대", "을지로", "종각", "서울역"]
        refs = ["GS주유소", "CU편의점", "세븐일레븐", "우리은행", "국민은행",
                "큰 아파트", "빨간 건물", "하얀 빌딩", "공원", "놀이터"]
        directions = ["왼쪽", "오른쪽", "직진"]
        distances = ["50미터", "100미터", "200미터", "한 블록"]
        rels = ["앞", "뒤", "옆", "건너편", "맞은편"]
        details = ["2층이요", "지하예요", "골목 안이요", "큰 건물이요"]
        roads = ["큰길", "대로", "이 도로"]
        exits = list(range(1, 9))

        template = random.choice(templates)
        text = template.format(
            landmark=random.choice(landmarks),
            station=random.choice(stations),
            exit=random.choice(exits),
            ref=random.choice(refs),
            direction=random.choice(directions),
            distance=random.choice(distances),
            rel=random.choice(rels),
            detail=random.choice(details),
            road=random.choice(roads),
        )

        hotwords = []
        return text, hotwords

    def random_highway(self) -> Tuple[str, List[str]]:
        """랜덤 고속도로 위치 생성"""
        hw_name = random.choice(list(self.highways.keys()))
        ics = self.highways[hw_name]
        ic = random.choice(ics)

        directions = ["서울방향", "부산방향", "광주방향", "대전방향", "상행", "하행"]
        distances = ["1km 전방", "2km 지점", "3km 전방", "500미터 앞", "직전"]

        addr = f"{hw_name} {random.choice(directions)} {ic} {random.choice(distances)}"
        hotwords = [hw_name, ic]
        return addr, hotwords

    def random_apartment(self) -> Tuple[str, List[str]]:
        """랜덤 아파트 주소 생성"""
        apts = ["래미안", "자이", "푸르지오", "힐스테이트", "아이파크",
                "롯데캐슬", "이편한세상", "더샵", "SK뷰", "주공아파트",
                "현대아파트", "삼성아파트", "대림아파트"]
        apt = random.choice(apts)

        dong = f"{random.randint(101, 115)}동"
        ho = f"{random.randint(1, 30)}0{random.randint(1,4)}호"

        addr_base, hotwords = self.random_road_address()
        addr = f"{addr_base} {apt} {dong} {ho}"
        hotwords.append(apt)
        return addr, hotwords

    def random_address(self) -> Tuple[str, str, List[str]]:
        """랜덤 주소를 유형과 함께 반환합니다."""
        addr_type = random.choices(
            ["road", "jibeon", "intersection", "landmark", "colloquial", "highway", "apartment"],
            weights=[30, 15, 15, 10, 15, 5, 10],
        )[0]

        func_map = {
            "road": self.random_road_address,
            "jibeon": self.random_jibeon_address,
            "intersection": self.random_intersection,
            "landmark": self.random_landmark,
            "colloquial": self.random_colloquial,
            "highway": self.random_highway,
            "apartment": self.random_apartment,
        }

        addr, hotwords = func_map[addr_type]()
        return addr_type, addr, hotwords


# ---------- 대화 스크립트 생성 ----------

class ScriptGenerator:
    """119 신고전화 대화 스크립트를 생성합니다."""

    EMERGENCIES = [
        ("화재", ["화재", "불", "연기"]),
        ("구급", ["구급", "쓰러짐", "의식 없음", "호흡곤란"]),
        ("교통사고", ["교통사고", "추돌", "사고"]),
        ("구조", ["구조", "갇힘", "끼임"]),
        ("가스누출", ["가스 누출", "가스 냄새"]),
    ]

    CALLER_OPENERS = [
        "여보세요 119죠? {emergency} 났어요!",
        "119요? 여기 {emergency} 신고합니다",
        "여기 {emergency}예요 빨리 와주세요",
        "119 {emergency} 신고요",
        "살려주세요 여기 {emergency}예요",
    ]

    DISPATCHER_ASK_ADDR = [
        "네 정확한 주소 말씀해 주세요",
        "네 위치가 어디신가요?",
        "주소 알려주세요",
        "네 어디세요?",
        "위치 확인 부탁드립니다",
    ]

    DISPATCHER_CONFIRM = [
        "{addr} 맞으시죠?",
        "{addr} 맞습니까?",
        "{addr} 확인합니다",
        "{addr}이요? 맞으시죠?",
    ]

    DISPATCHER_DISPATCH = [
        "소방대 출동합니다. 안전한 곳으로 대피해 주세요",
        "지금 바로 출동합니다. 현장에서 기다려 주세요",
        "출동 지시 내렸습니다. 위험한 곳에서 벗어나 주세요",
        "구급대 출동합니다. 환자 옆에 계세요",
    ]

    CALLER_DETAILS = [
        "네 빨리 와주세요",
        "네 여기 사람이 많아요",
        "지금 상황이 심각해요",
        "아 네 감사합니다",
        "빨리요 제발",
    ]

    def generate_script(
        self, addr_type: str, address: str, hotwords: List[str]
    ) -> Tuple[List[Dict], List[str]]:
        """주소를 포함한 119 대화 스크립트를 생성합니다."""
        emergency, emg_keywords = random.choice(self.EMERGENCIES)

        segments = []
        t = 0.0

        # 1. 신고자 오프닝
        opener = random.choice(self.CALLER_OPENERS).format(emergency=emergency)
        dur = max(2.0, len(opener) * 0.15)
        segments.append({"speaker": 0, "text": opener, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 2. 상황실 주소 요청
        ask = random.choice(self.DISPATCHER_ASK_ADDR)
        dur = max(1.5, len(ask) * 0.12)
        segments.append({"speaker": 1, "text": ask, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 3. 신고자 주소 말하기
        if addr_type == "colloquial":
            # 구어체는 그대로
            addr_text = address
        elif addr_type == "apartment":
            addr_text = f"여기 {address}요"
        else:
            prefixes = ["", "여기 ", "주소가 ", "여기 주소는 "]
            suffixes = ["요", "이요", "입니다", "에요"]
            addr_text = f"{random.choice(prefixes)}{address}{random.choice(suffixes)}"

        dur = max(3.0, len(addr_text) * 0.15)
        segments.append({"speaker": 0, "text": addr_text, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 4. 상황실 주소 확인 (반복)
        confirm = random.choice(self.DISPATCHER_CONFIRM).format(addr=address)
        dur = max(2.0, len(confirm) * 0.12)
        segments.append({"speaker": 1, "text": confirm, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 5. 신고자 확인
        yes_texts = ["네 맞아요", "네", "맞아요 맞아요", "네 그곳이요"]
        yes_text = random.choice(yes_texts)
        dur = max(1.0, len(yes_text) * 0.12)
        segments.append({"speaker": 0, "text": yes_text, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 6. 추가 정보 (선택)
        if random.random() < 0.5:
            floor = random.choice(["1층", "2층", "3층", "5층", "10층", "지하1층", "옥상"])
            detail_text = f"여기 {floor}이에요"
            dur = max(1.5, len(detail_text) * 0.12)
            segments.append({"speaker": 0, "text": detail_text, "start": round(t, 2), "end": round(t + dur, 2)})
            t += dur + 0.3

        # 7. 출동 안내
        dispatch = random.choice(self.DISPATCHER_DISPATCH)
        dur = max(2.0, len(dispatch) * 0.12)
        segments.append({"speaker": 1, "text": dispatch, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur + 0.3

        # 8. 마무리
        closer = random.choice(self.CALLER_DETAILS)
        dur = max(1.0, len(closer) * 0.12)
        segments.append({"speaker": 0, "text": closer, "start": round(t, 2), "end": round(t + dur, 2)})
        t += dur

        all_hotwords = list(set(hotwords + emg_keywords))
        return segments, all_hotwords


# ---------- 오디오 증강 ----------

def augment_phone_quality(audio: np.ndarray, sr: int) -> np.ndarray:
    """전화 환경 시뮬레이션: 대역 제한 + 노이즈 + 볼륨 변동"""

    # 1. 전화 대역 제한 (300Hz-3400Hz) - 간단한 FIR 근사
    # 저역 차단
    if sr > 300:
        n = len(audio)
        freq = np.fft.rfftfreq(n, 1.0 / sr)
        fft = np.fft.rfft(audio)
        fft[freq < 300] *= 0.1
        fft[freq > 3400] *= 0.1
        audio = np.fft.irfft(fft, n).astype(np.float32)

    # 2. 백색 노이즈 추가
    noise_level = random.uniform(0.002, 0.015)
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_level
    audio = audio + noise

    # 3. 볼륨 변동
    volume = random.uniform(0.6, 1.2)
    audio = audio * volume

    # 4. 약간의 클리핑 (전화 음질)
    audio = np.clip(audio, -0.95, 0.95)

    return audio


def add_background_noise(audio: np.ndarray, sr: int, noise_type: str = "random") -> np.ndarray:
    """배경 소음 추가"""
    if noise_type == "siren":
        # 사이렌 시뮬레이션 (사인파 변조)
        t = np.arange(len(audio)) / sr
        siren = np.sin(2 * np.pi * 800 * t + 200 * np.sin(2 * np.pi * 2 * t)) * 0.02
        audio = audio + siren.astype(np.float32)

    elif noise_type == "traffic":
        # 교통 소음 (저주파 노이즈)
        noise = np.random.randn(len(audio)).astype(np.float32)
        # 저주파만
        n = len(noise)
        freq = np.fft.rfftfreq(n, 1.0 / sr)
        fft = np.fft.rfft(noise)
        fft[freq > 500] *= 0.05
        noise = np.fft.irfft(fft, n).astype(np.float32)
        audio = audio + noise * 0.03

    elif noise_type == "crowd":
        # 군중 소음 (핑크 노이즈)
        noise = np.random.randn(len(audio)).astype(np.float32)
        audio = audio + noise * 0.01

    else:
        noise_type_choice = random.choice(["white", "none"])
        if noise_type_choice == "white":
            noise = np.random.randn(len(audio)).astype(np.float32) * 0.005
            audio = audio + noise

    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ---------- TTS 합성 ----------

async def synthesize_edge_tts(text: str, output_path: str, voice: str = None) -> bool:
    """edge-tts로 한국어 음성 합성"""
    try:
        import edge_tts
    except ImportError:
        logger.error("edge-tts가 설치되지 않았습니다: pip install edge-tts")
        return False

    if voice is None:
        voices = [
            "ko-KR-SunHiNeural",    # 여성
            "ko-KR-InJoonNeural",   # 남성
            "ko-KR-HyunsuNeural",   # 남성
        ]
        voice = random.choice(voices)

    rate = random.choice(["-10%", "-5%", "+0%", "+5%", "+10%"])
    pitch = random.choice(["-5Hz", "+0Hz", "+5Hz"])

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)
    return True


def save_wav(audio: np.ndarray, sr: int, path: str):
    """float32 오디오를 WAV로 저장"""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """WAV 파일을 float32로 로드"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


# ---------- 메인 생성기 ----------

async def generate_dataset(
    output_dir: str,
    count: int = 500,
    augment: bool = True,
    seed: int = 42,
):
    """주소 학습 데이터를 대량 생성합니다."""
    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    addr_gen = AddressGenerator()
    script_gen = ScriptGenerator()

    stats = {"generated": 0, "errors": 0}
    noise_types = ["random", "siren", "traffic", "crowd"]

    for i in range(count):
        try:
            # 1. 랜덤 주소 생성
            addr_type, address, hotwords = addr_gen.random_address()

            # 2. 대화 스크립트 생성
            segments, all_hotwords = script_gen.generate_script(addr_type, address, hotwords)

            # 3. 전체 텍스트
            full_text = " ".join(seg["text"] for seg in segments)

            # 4. TTS 합성
            call_id = f"addr_{addr_type}_{i:05d}"
            mp3_path = str(out_path / f"{call_id}_raw.mp3")

            ok = await synthesize_edge_tts(full_text, mp3_path)
            if not ok:
                stats["errors"] += 1
                continue

            # 5. MP3 → WAV 변환 + 증강
            wav_path = str(out_path / f"{call_id}.wav")

            try:
                # ffmpeg로 변환 (있으면)
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_path, "-ar", "24000", "-ac", "1", wav_path],
                    capture_output=True, timeout=10,
                )
                os.unlink(mp3_path)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                # ffmpeg 없으면 mp3 그대로 사용
                os.rename(mp3_path, wav_path)

            # 증강
            if augment and os.path.exists(wav_path):
                try:
                    audio, sr = load_wav(wav_path)
                    audio = augment_phone_quality(audio, sr)
                    noise = random.choice(noise_types)
                    audio = add_background_noise(audio, sr, noise)
                    save_wav(audio, sr, wav_path)
                    duration = len(audio) / sr
                except Exception:
                    duration = 0
            else:
                duration = 0

            # 6. JSON 레이블 저장
            label = {
                "audio_path": f"{call_id}.wav",
                "audio_duration": round(duration, 2),
                "segments": segments,
                "customized_context": all_hotwords,
                "metadata": {
                    "addr_type": addr_type,
                    "address": address,
                    "generated": True,
                    "augmented": augment,
                },
            }

            with open(out_path / f"{call_id}.json", "w", encoding="utf-8") as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            stats["generated"] += 1

            if (i + 1) % 50 == 0:
                logger.info(f"생성 진행: {i+1}/{count} ({stats['generated']}건 성공)")

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 10:
                logger.warning(f"생성 오류 [{i}]: {e}")

    logger.info(f"생성 완료: {stats['generated']}건 성공, {stats['errors']}건 오류")
    logger.info(f"출력 경로: {output_dir}")
    logger.info(f"임포트: python demo/realtime_asr/pipeline.py import --data-dir {output_dir}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="119 주소 음성 합성 데이터 생성")
    parser.add_argument("--output-dir", required=True, help="출력 디렉토리")
    parser.add_argument("--count", type=int, default=500, help="생성 건수")
    parser.add_argument("--no-augment", action="store_true", help="노이즈 증강 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    asyncio.run(generate_dataset(
        output_dir=args.output_dir,
        count=args.count,
        augment=not args.no_augment,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
