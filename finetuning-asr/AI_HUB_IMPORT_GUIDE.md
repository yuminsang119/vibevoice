# AI Hub 주소/위치 데이터 임포트 가이드

## 추천 데이터셋 (주소 WER 공략용)

### 필수 데이터셋

| 데이터셋 | AI Hub ID | 규모 | 용도 | 우선순위 |
|----------|-----------|------|------|----------|
| 긴급 상황 신고 음성 | - | ~1,000시간 | 119/112 실제 패턴 | ★★★ |
| 한국어 음성 (일반) | 71265 | ~1,000시간 | 주소 발화 포함 | ★★★ |
| 전화망 음성인식 | 71330 | ~600시간 | 전화 품질 대응 | ★★★ |
| 소음환경 음성인식 | 71319 | ~500시간 | 현장 소음 대응 | ★★☆ |

### 주소 특화 데이터셋

| 데이터셋 | AI Hub ID | 규모 | 용도 | 우선순위 |
|----------|-----------|------|------|----------|
| 교통정보 음성 | - | ~300시간 | 도로/교차로 명칭 | ★★★ |
| 내비게이션 음성명령 | - | ~150시간 | 주소 발화 패턴 | ★★☆ |
| 한국어 지명 음성 | - | ~200시간 | 지역명 인식 | ★★☆ |

### 방언/특수화자 데이터셋

| 데이터셋 | AI Hub ID | 규모 | 용도 | 우선순위 |
|----------|-----------|------|------|----------|
| 방언 발화 (경상) | 71506 | ~200시간 | 경상도 신고자 | ★★☆ |
| 방언 발화 (전라) | 71507 | ~200시간 | 전라도 신고자 | ★★☆ |
| 방언 발화 (충청) | 71508 | ~200시간 | 충청도 신고자 | ★★☆ |
| 방언 발화 (제주) | 71509 | ~200시간 | 제주도 신고자 | ★☆☆ |
| 고령자 음성 | 71376 | ~500시간 | 노인 신고자 | ★★★ |
| 아동 음성 | - | ~200시간 | 어린이 신고자 | ★☆☆ |

---

## 다운로드 후 전처리

### 1. AI Hub 데이터 구조 (일반적)

```
aihub_dataset/
├── 1.Training/
│   ├── 라벨링데이터/
│   │   └── *.json (전사 텍스트)
│   └── 원천데이터/
│       └── *.wav (오디오)
└── 2.Validation/
    ├── 라벨링데이터/
    └── 원천데이터/
```

### 2. 전처리 스크립트

```bash
# AI Hub → 파이프라인 형식 변환
python finetuning-asr/preprocess_aihub.py \
    --input-dir /data/aihub/긴급상황신고음성 \
    --output-dir /data/processed/emergency \
    --format aihub_emergency

# 파이프라인에 임포트 (사람 검수 완료 데이터)
python demo/realtime_asr/pipeline.py import \
    --data-dir /data/processed/emergency
```

### 3. AI Hub JSON → VibeVoice 형식 변환 규칙

AI Hub 일반 형식:
```json
{
  "id": "KsponSpeech_000001",
  "metadata": {"title": "...", "speaker": {"id": "S001", "age": 30}},
  "utterance": [
    {"id": "u0001", "form": "안녕하세요", "start": 0.0, "end": 1.2}
  ]
}
```

VibeVoice 학습 형식 (변환 결과):
```json
{
  "audio_path": "KsponSpeech_000001.wav",
  "audio_duration": 30.5,
  "segments": [
    {"speaker": 0, "text": "안녕하세요", "start": 0.0, "end": 1.2}
  ],
  "customized_context": ["관련 핫워드"]
}
```

---

## 주소 데이터 필터링

AI Hub 대규모 데이터에서 주소 관련 발화만 추출:

```bash
# 주소 패턴이 포함된 발화만 필터링
python finetuning-asr/preprocess_aihub.py \
    --input-dir /data/aihub/한국어음성 \
    --output-dir /data/processed/address_only \
    --filter-address  # 주소 패턴 포함 발화만 추출
```

필터링 키워드: `로`, `길`, `동`, `번지`, `아파트`, `사거리`, `교차로`, `IC`, `구`, `시`

---

## 학습 순서 권장

```
Phase 1: 보유 1600h + AI Hub 일반 한국어
         → 기본 WER 하락 (9.65% → 5~6%)

Phase 2: AI Hub 긴급상황 + 전화망 + 소음환경
         → 도메인 적응 (5~6% → 3~4%)

Phase 3: 주소 TTS 합성 + AI Hub 교통정보 + 지명
         → 주소 WER 집중 (주소WER 12% → 3~4%)

Phase 4: 방언 + 고령자 + 아동
         → 극한 상황 (전체 WER 2~3%)

Phase 5: 지속 학습 (신규 통화 자동 수집)
         → 수렴 (전체 WER 2% 이하)
```
