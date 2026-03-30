#!/bin/bash
# 119 신고전화 ASR 파인튜닝 실행 스크립트
#
# 사용법:
#   bash finetuning-asr/train_119.sh
#
# 사전 준비:
#   1. 119_dataset/ 에 실제 오디오 파일(.wav)을 배치
#   2. JSON 레이블 파일의 audio_path가 오디오 파일명과 일치하는지 확인
#   3. address_hotwords.json의 핫워드가 customized_context에 포함되어 있는지 확인

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/119_dataset"
OUTPUT_DIR="${SCRIPT_DIR}/output_119_asr"

echo "============================================"
echo "  119 신고전화 ASR LoRA 파인튜닝"
echo "============================================"
echo ""
echo "데이터 디렉토리: ${DATA_DIR}"
echo "출력 디렉토리:   ${OUTPUT_DIR}"
echo ""

# 데이터셋 파일 수 확인
NUM_JSON=$(find "${DATA_DIR}" -name "call_*.json" | wc -l)
NUM_AUDIO=$(find "${DATA_DIR}" -name "*.wav" -o -name "*.mp3" | wc -l)
echo "JSON 레이블: ${NUM_JSON}개"
echo "오디오 파일: ${NUM_AUDIO}개"
echo ""

if [ "${NUM_AUDIO}" -eq 0 ]; then
    echo "[경고] 오디오 파일이 없습니다."
    echo "119_dataset/ 디렉토리에 실제 통화 녹음 파일을 추가해 주세요."
    echo "JSON의 audio_path 필드와 파일명이 일치해야 합니다."
    echo ""
    echo "예시:"
    echo "  119_dataset/119_call_001.wav  (call_fire_001.json의 audio_path)"
    echo "  119_dataset/119_call_002.wav  (call_medical_001.json의 audio_path)"
    echo ""
    exit 1
fi

python "${SCRIPT_DIR}/lora_finetune.py" \
    --model_path microsoft/VibeVoice-ASR \
    --data_dir "${DATA_DIR}" \
    --use_customized_context True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_steps 50 \
    --logging_steps 5 \
    --save_steps 50 \
    --save_total_limit 3 \
    --bf16 True \
    --seed 42 \
    --report_to none

echo ""
echo "============================================"
echo "  학습 완료!"
echo "  모델 저장 위치: ${OUTPUT_DIR}"
echo "============================================"
