#!/bin/bash
# =============================================================================
#  119 ASR 최적 학습 설정 - Server A (H100 x4, 320GB VRAM)
#
#  목표: WER을 최대한 빠르고 낮게 떨어뜨리기 위한 세팅
#
#  사용법:
#    bash finetuning-asr/train_119_h100x4.sh
#
#    # 데이터 경로 지정
#    DATA_DIR=/data/1600h bash finetuning-asr/train_119_h100x4.sh
#
#    # 이전 모델에서 이어서 학습
#    RESUME_MODEL=./pipeline_data/models/119_asr_v0003 bash finetuning-asr/train_119_h100x4.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------- 경로 설정 ----------
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/119_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_119_h100x4}"
MODEL_PATH="${RESUME_MODEL:-microsoft/VibeVoice-ASR}"
NUM_GPUS="${NUM_GPUS:-4}"

echo "============================================================"
echo "  119 ASR 최적 학습 - H100 x${NUM_GPUS}"
echo "============================================================"
echo ""
echo "  모델:    ${MODEL_PATH}"
echo "  데이터:  ${DATA_DIR}"
echo "  출력:    ${OUTPUT_DIR}"
echo "  GPU:     ${NUM_GPUS}장"
echo ""

# ---------- 데이터 확인 ----------
NUM_JSON=$(find "${DATA_DIR}" -name "*.json" -not -name "*hotwords*" | wc -l)
echo "  학습 데이터: ${NUM_JSON}건"
echo ""

if [ "${NUM_JSON}" -eq 0 ]; then
    echo "[오류] 학습 데이터가 없습니다. DATA_DIR을 확인하세요."
    exit 1
fi

# ---------- 데이터 규모별 하이퍼파라미터 자동 조정 ----------
# 1600시간 = ~50,000건 기준
if [ "${NUM_JSON}" -lt 100 ]; then
    # 소규모: 높은 에포크, 작은 LR
    EPOCHS=20
    LR="5e-5"
    WARMUP=100
    LORA_R=16
    LORA_ALPHA=32
    BATCH_PER_GPU=1
    GRAD_ACCUM=8
    SAVE_STEPS=50
    echo "  모드: 소규모 (<100건) → 에포크 ${EPOCHS}, LR ${LR}"
elif [ "${NUM_JSON}" -lt 1000 ]; then
    # 중규모: 적당한 에포크
    EPOCHS=10
    LR="1e-4"
    WARMUP=200
    LORA_R=32
    LORA_ALPHA=64
    BATCH_PER_GPU=1
    GRAD_ACCUM=8
    SAVE_STEPS=100
    echo "  모드: 중규모 (100~1000건) → 에포크 ${EPOCHS}, LR ${LR}, LoRA r=${LORA_R}"
elif [ "${NUM_JSON}" -lt 10000 ]; then
    # 대규모: 낮은 에포크, 큰 LoRA rank
    EPOCHS=5
    LR="1e-4"
    WARMUP=500
    LORA_R=64
    LORA_ALPHA=128
    BATCH_PER_GPU=1
    GRAD_ACCUM=4
    SAVE_STEPS=500
    echo "  모드: 대규모 (1K~10K건) → 에포크 ${EPOCHS}, LR ${LR}, LoRA r=${LORA_R}"
else
    # 초대규모: 1~2 에포크로 충분
    EPOCHS=3
    LR="5e-5"
    WARMUP=1000
    LORA_R=64
    LORA_ALPHA=128
    BATCH_PER_GPU=1
    GRAD_ACCUM=4
    SAVE_STEPS=1000
    echo "  모드: 초대규모 (10K건+) → 에포크 ${EPOCHS}, LR ${LR}, LoRA r=${LORA_R}"
fi

# 환경변수로 오버라이드 가능
EPOCHS="${TRAIN_EPOCHS:-$EPOCHS}"
LR="${TRAIN_LR:-$LR}"
LORA_R="${TRAIN_LORA_R:-$LORA_R}"
LORA_ALPHA="${TRAIN_LORA_ALPHA:-$LORA_ALPHA}"

# 유효 배치 사이즈 = BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM
EFFECTIVE_BATCH=$((BATCH_PER_GPU * NUM_GPUS * GRAD_ACCUM))
echo "  유효 배치: ${EFFECTIVE_BATCH} (${BATCH_PER_GPU} x ${NUM_GPUS}GPU x ${GRAD_ACCUM}accum)"
echo ""

# ---------- H100 최적화 환경 변수 ----------
# NCCL 최적화 (H100 NVLink/NVSwitch)
export NCCL_P2P_LEVEL=NVL        # NVLink P2P 활성화
export NCCL_IB_DISABLE=0         # InfiniBand 활성화 (있으면)
export NCCL_NET_GDR_LEVEL=5      # GPU Direct RDMA

# CUDA 최적화
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 메모리 효율
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# TF32 활성화 (H100에서 bf16만큼 빠르면서 정밀도 향상)
export NVIDIA_TF32_OVERRIDE=1

# Flash Attention 2 강제 사용
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# PyTorch 최적화
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "  학습 시작..."
echo "============================================================"
echo ""

# ---------- 멀티 GPU 학습 실행 ----------
# accelerate 사용 (DDP: Distributed Data Parallel)
# H100 x4에서는 DDP가 가장 효율적 (모델 7B = ~14GB, VRAM 80GB/GPU 여유)

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    "${SCRIPT_DIR}/lora_finetune.py" \
    --model_path "${MODEL_PATH}" \
    --data_dir "${DATA_DIR}" \
    --use_customized_context True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_PER_GPU} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --lr_scheduler_type cosine \
    --warmup_steps ${WARMUP} \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 5 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --seed 42 \
    --report_to none \
    --ddp_find_unused_parameters False \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout 0.05 \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "============================================================"
echo "  학습 완료!"
echo "  모델: ${OUTPUT_DIR}"
echo "============================================================"
