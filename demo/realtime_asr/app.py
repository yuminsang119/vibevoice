"""
실시간 전화 음성 → ASR 변환 + 주소 좌표 변환 서버

실시간으로 전화 음성(PCM 오디오 스트림)을 WebSocket으로 수신하고,
VibeVoice ASR 모델로 텍스트 변환 후 주소를 추출하여 좌표로 변환합니다.

사용법:
    MODEL_PATH=microsoft/VibeVoice-ASR python app.py

    # 카카오 지오코딩 사용 시 (권장)
    KAKAO_REST_API_KEY=... MODEL_PATH=microsoft/VibeVoice-ASR python app.py

클라이언트 연결:
    ws://localhost:8080/ws/transcribe
    - 오디오 청크를 binary 메시지로 전송 (16-bit PCM, 16kHz, mono)
    - 전사 결과 + 주소 좌표를 JSON text 메시지로 수신

Twilio 연동:
    ws://localhost:8080/ws/twilio
    - Twilio Media Stream 프로토콜과 호환

REST API:
    GET /api/geocode?address=서울시+강남구+테헤란로+152
    POST /api/extract-address  {"text": "서울 강남구 테헤란로 152 강남파이낸스센터"}
"""

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import urllib.parse

import numpy as np
import torch
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from geocoding import extract_addresses, geocode, geocode_from_segments, GeoResult
from control import SessionManager, classify_urgency, classify_report_type, extract_patient_info

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ASR 입력 샘플레이트
ASR_SAMPLE_RATE = 24000
# 전화 음성 기본 샘플레이트
PHONE_SAMPLE_RATE = 16000
# 청크 버퍼 시간 (초) - 이 시간만큼 모이면 ASR 실행
CHUNK_BUFFER_SECONDS = 5.0
# 최대 오디오 길이 (초)
MAX_AUDIO_SECONDS = 600


def get_timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """16-bit PCM bytes를 float32 numpy 배열로 변환"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def mulaw_decode(data: bytes) -> np.ndarray:
    """mu-law 인코딩된 바이트를 float32로 디코딩 (Twilio 형식)"""
    MULAW_BIAS = 33
    MULAW_MAX = 0x1FFF

    result = np.zeros(len(data), dtype=np.float32)
    for i, byte in enumerate(data):
        byte = ~byte
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        sample = (mantissa << 3) + MULAW_BIAS
        sample <<= exponent
        sample -= MULAW_BIAS
        if sign:
            sample = -sample
        result[i] = sample / 32768.0
    return result


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """간단한 리샘플링 (선형 보간)"""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def save_audio_to_tempfile(audio: np.ndarray, sample_rate: int) -> str:
    """오디오를 임시 WAV 파일로 저장"""
    import wave
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


class RealtimeASRService:
    """실시간 ASR 서비스 - VibeVoice ASR 모델을 사용한 음성 인식"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.hotwords: List[str] = []

    def load(self):
        """모델과 프로세서 로드"""
        from vibevoice.modular.modeling_vibevoice_asr import (
            VibeVoiceASRForConditionalGeneration,
        )
        from vibevoice.processor.vibevoice_asr_processor import (
            VibeVoiceASRProcessor,
        )

        logger.info(f"ASR 모델 로드 중: {self.model_path}")

        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn_impl = "flash_attention_2" if self.device == "cuda" else "sdpa"

        try:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
        except Exception:
            logger.warning("flash_attention_2 실패, sdpa로 대체")
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device if self.device != "cpu" else None,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        logger.info("ASR 모델 로드 완료")

    def set_hotwords(self, hotwords: List[str]):
        """핫워드 설정 (119 관련 키워드)"""
        self.hotwords = hotwords
        logger.info(f"핫워드 설정: {hotwords}")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """오디오 파일을 전사"""
        if self.model is None or self.processor is None:
            raise RuntimeError("모델이 로드되지 않았습니다")

        context_info = None
        if self.hotwords:
            context_info = "\n".join(self.hotwords)

        encoding = self.processor._process_single_audio(
            audio_path,
            sampling_rate=None,
            add_generation_prompt=True,
            use_streaming=True,
            context_info=context_info,
        )

        # 모델 입력 준비
        input_ids = torch.tensor([encoding["input_ids"]], device=self.device)
        acoustic_mask = torch.tensor(
            [encoding["acoustic_input_mask"]], device=self.device
        )
        speech = torch.tensor(
            encoding["speech"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                acoustic_input_mask=acoustic_mask,
                speech_tensors=speech,
                max_new_tokens=32768,
                temperature=0.0,
            )

        # 디코딩
        generated_ids = outputs[0][input_ids.shape[1] :]
        text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # JSON 파싱 시도
        try:
            segments = json.loads(text)
        except json.JSONDecodeError:
            segments = [{"Content": text}]

        return {
            "timestamp": get_timestamp(),
            "segments": segments,
            "raw_text": text,
        }


async def enrich_with_geocoding(result: Dict[str, Any]) -> Dict[str, Any]:
    """ASR 결과에 주소 추출 + 좌표 변환 결과를 추가합니다."""
    segments = result.get("segments", [])
    if segments:
        try:
            locations = await geocode_from_segments(segments)
            if locations:
                result["locations"] = locations
                logger.info(f"주소 {len(locations)}건 추출/변환 완료")
        except Exception as e:
            logger.warning(f"지오코딩 실패: {e}")
    return result


app = FastAPI(title="VibeVoice 실시간 ASR + 지오코딩 서버")


@app.on_event("startup")
async def startup():
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH 환경변수를 설정하세요")

    device = os.environ.get("MODEL_DEVICE", "cuda")

    service = RealtimeASRService(model_path=model_path, device=device)
    service.load()

    # 119 신고전화용 기본 핫워드
    default_hotwords = os.environ.get("HOTWORDS", "")
    if default_hotwords:
        service.set_hotwords(default_hotwords.split(","))

    app.state.asr_service = service
    app.state.lock = asyncio.Lock()
    app.state.session_manager = SessionManager()
    logger.info("실시간 ASR 서버 준비 완료")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    """
    범용 실시간 ASR WebSocket 엔드포인트

    클라이언트가 16-bit PCM (16kHz, mono) 오디오 청크를 binary로 전송하면,
    일정 시간 버퍼링 후 ASR 결과를 JSON으로 반환합니다.

    Query params:
        - sample_rate: 오디오 샘플레이트 (기본: 16000)
        - buffer_seconds: 버퍼링 시간 (기본: 5.0)
        - hotwords: 핫워드 (쉼표 구분)
    """
    await ws.accept()
    logger.info("클라이언트 연결됨")

    sample_rate = int(ws.query_params.get("sample_rate", PHONE_SAMPLE_RATE))
    buffer_seconds = float(ws.query_params.get("buffer_seconds", CHUNK_BUFFER_SECONDS))
    hotwords_param = ws.query_params.get("hotwords", "")

    service: RealtimeASRService = app.state.asr_service
    if hotwords_param:
        service.set_hotwords(hotwords_param.split(","))

    audio_buffer = bytearray()
    buffer_size = int(sample_rate * 2 * buffer_seconds)  # 16-bit = 2 bytes/sample
    total_audio = bytearray()

    try:
        while True:
            data = await ws.receive()

            if data.get("type") == "websocket.disconnect":
                break

            if "bytes" in data and data["bytes"]:
                chunk = data["bytes"]
                audio_buffer.extend(chunk)
                total_audio.extend(chunk)

                # 버퍼가 충분히 차면 ASR 실행
                if len(audio_buffer) >= buffer_size:
                    audio_float = pcm16_to_float32(bytes(audio_buffer))
                    audio_resampled = resample_audio(
                        audio_float, sample_rate, ASR_SAMPLE_RATE
                    )

                    temp_path = save_audio_to_tempfile(
                        audio_resampled, ASR_SAMPLE_RATE
                    )
                    try:
                        result = await asyncio.to_thread(
                            service.transcribe, temp_path
                        )
                        result["type"] = "partial"
                        result = await enrich_with_geocoding(result)
                        await ws.send_text(json.dumps(result, ensure_ascii=False))
                    except Exception as e:
                        logger.error(f"ASR 오류: {e}")
                        await ws.send_text(
                            json.dumps({"type": "error", "message": str(e)})
                        )
                    finally:
                        os.unlink(temp_path)

                    audio_buffer.clear()

            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                cmd = msg.get("command")

                if cmd == "stop":
                    # 남은 버퍼 처리
                    if len(audio_buffer) > 0:
                        audio_float = pcm16_to_float32(bytes(audio_buffer))
                        audio_resampled = resample_audio(
                            audio_float, sample_rate, ASR_SAMPLE_RATE
                        )
                        temp_path = save_audio_to_tempfile(
                            audio_resampled, ASR_SAMPLE_RATE
                        )
                        try:
                            result = await asyncio.to_thread(
                                service.transcribe, temp_path
                            )
                            result["type"] = "final"
                            result = await enrich_with_geocoding(result)
                            await ws.send_text(
                                json.dumps(result, ensure_ascii=False)
                            )
                        finally:
                            os.unlink(temp_path)
                        audio_buffer.clear()
                    break

                elif cmd == "set_hotwords":
                    new_hotwords = msg.get("hotwords", [])
                    service.set_hotwords(new_hotwords)
                    await ws.send_text(
                        json.dumps({"type": "info", "message": f"핫워드 설정: {new_hotwords}"})
                    )

    except WebSocketDisconnect:
        logger.info("클라이언트 연결 해제")
    finally:
        # 전체 오디오에 대한 최종 전사
        if len(total_audio) > 0:
            audio_float = pcm16_to_float32(bytes(total_audio))
            audio_resampled = resample_audio(
                audio_float, sample_rate, ASR_SAMPLE_RATE
            )
            temp_path = save_audio_to_tempfile(audio_resampled, ASR_SAMPLE_RATE)
            try:
                result = await asyncio.to_thread(service.transcribe, temp_path)
                result["type"] = "final_complete"
                result = await enrich_with_geocoding(result)
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps(result, ensure_ascii=False))
            except Exception as e:
                logger.error(f"최종 전사 오류: {e}")
            finally:
                os.unlink(temp_path)

        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()
        logger.info("WebSocket 종료")


@app.websocket("/ws/twilio")
async def websocket_twilio(ws: WebSocket):
    """
    Twilio Media Stream 연동 엔드포인트

    Twilio의 <Stream> TwiML에서 이 엔드포인트로 연결하면,
    실시간 전화 음성을 ASR로 변환합니다.

    TwiML 예시:
        <Response>
            <Connect>
                <Stream url="wss://your-server.com/ws/twilio" />
            </Connect>
        </Response>
    """
    await ws.accept()
    logger.info("Twilio 스트림 연결됨")

    service: RealtimeASRService = app.state.asr_service
    stream_sid = None
    call_sid = None
    audio_buffer = bytearray()
    # Twilio는 8kHz mu-law
    twilio_sample_rate = 8000
    buffer_size = int(twilio_sample_rate * CHUNK_BUFFER_SECONDS)

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            event = msg.get("event")

            if event == "connected":
                logger.info("Twilio 미디어 스트림 연결 확인")

            elif event == "start":
                start_data = msg.get("start", {})
                stream_sid = start_data.get("streamSid")
                call_sid = start_data.get("callSid")
                logger.info(f"통화 시작 - CallSid: {call_sid}, StreamSid: {stream_sid}")

                # 119 신고전화 기본 핫워드 설정
                if not service.hotwords:
                    service.set_hotwords([
                        "화재", "구급", "구조", "교통사고",
                        "서울", "부산", "대구", "인천", "광주", "대전", "울산",
                        "아파트", "빌딩", "주택", "도로",
                    ])

            elif event == "media":
                media = msg.get("media", {})
                payload = media.get("payload", "")
                audio_bytes = base64.b64decode(payload)

                # mu-law를 float32로 디코딩
                audio_float = mulaw_decode(audio_bytes)
                # float32를 int16 PCM으로
                pcm = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
                audio_buffer.extend(pcm.tobytes())

                if len(audio_buffer) >= buffer_size * 2:  # 2 bytes per sample
                    audio_np = pcm16_to_float32(bytes(audio_buffer))
                    audio_resampled = resample_audio(
                        audio_np, twilio_sample_rate, ASR_SAMPLE_RATE
                    )
                    temp_path = save_audio_to_tempfile(
                        audio_resampled, ASR_SAMPLE_RATE
                    )
                    try:
                        result = await asyncio.to_thread(
                            service.transcribe, temp_path
                        )
                        result["type"] = "transcription"
                        result["call_sid"] = call_sid
                        result["stream_sid"] = stream_sid
                        result = await enrich_with_geocoding(result)
                        await ws.send_text(json.dumps(result, ensure_ascii=False))
                    except Exception as e:
                        logger.error(f"Twilio ASR 오류: {e}")
                    finally:
                        os.unlink(temp_path)

                    audio_buffer.clear()

            elif event == "stop":
                logger.info(f"통화 종료 - CallSid: {call_sid}")

                # 남은 버퍼 처리
                if len(audio_buffer) > 0:
                    audio_np = pcm16_to_float32(bytes(audio_buffer))
                    audio_resampled = resample_audio(
                        audio_np, twilio_sample_rate, ASR_SAMPLE_RATE
                    )
                    temp_path = save_audio_to_tempfile(
                        audio_resampled, ASR_SAMPLE_RATE
                    )
                    try:
                        result = await asyncio.to_thread(
                            service.transcribe, temp_path
                        )
                        result["type"] = "final"
                        result["call_sid"] = call_sid
                        result = await enrich_with_geocoding(result)
                        await ws.send_text(json.dumps(result, ensure_ascii=False))
                    except Exception as e:
                        logger.error(f"최종 전사 오류: {e}")
                    finally:
                        os.unlink(temp_path)
                break

    except WebSocketDisconnect:
        logger.info("Twilio 스트림 연결 해제")
    finally:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()


@app.get("/")
async def index():
    """테스트 페이지"""
    return FileResponse(Path(__file__).parent / "index.html")


@app.get("/api/geocode")
async def api_geocode(address: str = Query(..., description="변환할 주소")):
    """주소 → 좌표 변환 REST API"""
    result = await geocode(address)
    if result:
        return {
            "status": "ok",
            "result": result.to_dict(),
            "map_urls": {
                "kakao": f"https://map.kakao.com/link/map/{urllib.parse.quote(result.road_address or address)},{result.latitude},{result.longitude}",
                "naver": f"https://map.naver.com/v5/search/{urllib.parse.quote(result.road_address or address)}?c={result.longitude},{result.latitude},15,0,0,0,dh",
                "google": f"https://www.google.com/maps/search/?api=1&query={result.latitude},{result.longitude}",
            },
        }
    return {"status": "not_found", "message": f"주소를 찾을 수 없습니다: {address}"}


class ExtractAddressRequest(BaseModel):
    text: str


@app.post("/api/extract-address")
async def api_extract_address(req: ExtractAddressRequest):
    """텍스트에서 주소 추출 + 좌표 변환 REST API"""
    addresses = extract_addresses(req.text)
    results = []
    for addr_info in addresses:
        geo = await geocode(addr_info["full"])
        results.append({
            "extracted": addr_info,
            "geocode": geo.to_dict() if geo else None,
        })
    return {"status": "ok", "addresses": results, "count": len(results)}


# ---------- 통제(관제) API ----------

@app.get("/api/control/stats")
async def api_control_stats():
    """전체 현황 통계"""
    sm: SessionManager = app.state.session_manager
    return sm.get_stats()


@app.get("/api/control/sessions")
async def api_control_sessions(status: Optional[str] = None):
    """세션 목록 조회 (status: active, completed, dispatched)"""
    sm: SessionManager = app.state.session_manager
    if status == "active":
        return {"sessions": sm.get_active_sessions()}
    return {"sessions": sm.get_all_sessions()}


@app.get("/api/control/session/{session_id}")
async def api_control_session_detail(session_id: str):
    """세션 상세 조회"""
    sm: SessionManager = app.state.session_manager
    session = sm.get_session(session_id)
    if not session:
        return {"status": "not_found"}
    return session.to_dict()


@app.post("/api/control/session/{session_id}/dispatch")
async def api_control_dispatch(session_id: str):
    """출동 지령서 생성"""
    sm: SessionManager = app.state.session_manager
    dispatch = sm.generate_dispatch_order(session_id)
    if not dispatch:
        return {"status": "not_found"}
    return {"status": "ok", "dispatch_order": dispatch}


class AnalyzeTextRequest(BaseModel):
    text: str


@app.post("/api/control/analyze")
async def api_control_analyze(req: AnalyzeTextRequest):
    """텍스트 분석 (긴급도 + 신고유형 + 환자정보 + 주소)"""
    urgency = classify_urgency(req.text)
    report_type = classify_report_type(req.text)
    patients = extract_patient_info(req.text)
    addresses = extract_addresses(req.text)

    locations = []
    for addr in addresses:
        geo = await geocode(addr["full"])
        locations.append({
            "extracted": addr,
            "geocode": geo.to_dict() if geo else None,
        })

    return {
        "urgency": urgency,
        "report_type": report_type,
        "patients": patients,
        "locations": locations,
    }


@app.post("/api/control/session")
async def api_control_create_session():
    """수동 세션 생성"""
    sm: SessionManager = app.state.session_manager
    session = sm.create_session()
    return {"status": "ok", "session_id": session.session_id}


@app.post("/api/control/session/{session_id}/close")
async def api_control_close_session(session_id: str):
    """세션 종료"""
    sm: SessionManager = app.state.session_manager
    sm.close_session(session_id)
    return {"status": "ok"}


@app.get("/control")
async def control_page():
    """통제 대시보드 페이지"""
    return FileResponse(Path(__file__).parent / "control.html")


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": get_timestamp()}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
