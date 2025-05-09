#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

FSDP_CONFIG="configs/fsdp_config.yaml"

# default parameters
DEFAULT_EPOCHS=3
DEFAULT_BATCH_SIZE=16
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4

# fixed parameters
LR=5e-5
WARMUP_STEPS=100

MODEL_PATH=${1}
DATA_PATH=${2}
PAD_MODE=${3}
MAX_LENGTH=${4}
OUTPUT_DIR=${5}
BATCH_SIZE=${6:-$DEFAULT_BATCH_SIZE}
GRAD_ACCU=${7:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
EPOCHS=${8:-$DEFAULT_EPOCHS}

echo "Running training with parameters:"
echo "--------------------------------"
echo "Model Path: ${MODEL_PATH}"
echo "Data Path: ${DATA_PATH}"
echo "Pad Mode: ${PAD_MODE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation Steps: ${GRAD_ACCU}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LR}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "--------------------------------"

accelerate launch --config_file ${FSDP_CONFIG} --main_process_port 20001 sft/finetune.py \
  --model_name_or_path ${MODEL_PATH} \
  --data_path ${DATA_PATH} \
  --pad_mode ${PAD_MODE} \
  --max_length ${MAX_LENGTH} \
  --output_dir ${OUTPUT_DIR} \
  --overwrite_output_dir True \
  --gradient_checkpointing True \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCU} \
  --learning_rate ${LR} \
  --lr_scheduler_type "cosine" \
  --warmup_steps ${WARMUP_STEPS} \
  --save_strategy "no" \
  --logging_strategy "epoch" \
  --bf16 True \
  --tf32 True
