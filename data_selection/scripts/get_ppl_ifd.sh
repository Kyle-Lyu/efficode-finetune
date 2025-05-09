model="deepseek-ai/deepseek-coder-6.7b-base"

index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 11512))
  end_index=$(((i + 1) * 11512))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
      CUDA_VISIBLE_DEVICES=$gpu python data_selection/get_ppl_ifd.py \
            --model_path ${model} \
            --data_path "data/train/Mixed-Python-92k.jsonl" \
            --save_path "data/ppl_ifd/ds-6.7b/split/mixed_analysis_${i}.jsonl" \
            --start $start_index --end $end_index \
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

wait

sleep 3
python data_selection/merge_analysis.py \
    --input_dir "data/ppl_ifd/ds-6.7b/split/" \
    --output_file "data/ppl_ifd/ds-6.7b/mixed_analysis.jsonl"
