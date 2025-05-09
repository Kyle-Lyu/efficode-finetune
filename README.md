# Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection

## Overview

Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. 

Experimental results demonstrate that using only 10K samples, our method achieved 69.5\% on HumanEval and 77.2\% on MBPP, surpassing full-data training by 2.4\% and 2.3\%, respectively. Additionally, our method outperforms other sampling approaches across various data scales while requiring minimal sampling time. This underscores that our method effectively boosts model performance while significantly reducing computational costs.

## Usage

### Installation

Ensure that CUDA 12.1 is installed and then follow the commands below to set up the environment:

```bash
conda create -n finetune python=3.10 -y
conda activate finetune

pip install -r requirements.txt
```

### Data Selection

Data is formatted as follows:

```json
{
    "id": 1,
    "instruction": "Hi, there.",
    "response": "Hello! How can I help you today?"
}
```

Obtain embeddings with [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model:

```bash
python data_selection/get_embeddings.py \
--data_path data/train/Mixed-Python-92k.jsonl \
--save_path data/train/Mixed-Python-92k.pt \
--model_path sentence-transformers/all-mpnet-base-v2
```

Incorporate the parametric model into the code data selection process:

```bash
python data_selection/ours/select.py \
--data_path data/train/Mixed-Python-92k.jsonl \
--pt_path data/train/Mixed-Python-92k.pt \
--output_dir data/train \
--sample_num 10000
```

Sampled training data is saved in `data/train/Mixed-Python-92k-sn10K.jsonl`.

<details>
  <summary>Usage of Other Data Selection Methods</summary>

#### IFD/PPL

Compute IFD/PPL scores and select data accordingly.

```bash
# compute IFD/PPL scores
bash data_selection/scripts/get_ppl_ifd.sh

# select data by IFD scores
python data_selection/ifd/ifd_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--score_path "data/ppl_ifd/ds-6.7b/mixed_analysis.jsonl" \
--sample_num 10000 \
--output_dir "data/train/ifd"

# select data by PPL scores
python data_selection/ppl/ppl_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--score_path "data/ppl_ifd/ds-6.7b/mixed_analysis.jsonl" \
--sample_num 10000 \
--output_dir "data/train/ppl"
```

#### DEITA

Calculate the complexity and quality scores and then select data.

```bash
# compute the complexity scores
python data_selection/deita/get_complexity_scores.py \
--data_path data/train/Mixed-Python-92k.jsonl \
--save_path data/deita/mixed_complexity_scores.jsonl \
--model_path hkust-nlp/deita-complexity-scorer

# compute the quality scores
python data_selection/deita/get_quality_scores.py \
--data_path data/train/Mixed-Python-92k.jsonl \
--save_path data/deita/mixed_quality_scores.jsonl \
--model_path hkust-nlp/deita-quality-scorer

python data_selection/deita/get_analysis.py \
--complexity_file data/deita/mixed_complexity_scores.jsonl \
--quality_file data/deita/mixed_quality_scores.jsonl \
--output_file data/deita/mixed_analysis.jsonl

# select data
python data_selection/deita/deita_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--analysis_path "data/deita/mixed_analysis.jsonl" \
--pt_path "data/train/Mixed-Python-92k.pt" \
--output_dir "data/train/deita" \
--sample_num 10000
```

#### DQ

Partition the dataset into different bins and then select data.

```bash
# partition the dataset into different bins
python data_selection/dq/get_bins.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--pt_path "data/train/Mixed-Python-92k.pt" \
--output_dir "data/dq"

# select data
python data_selection/dq/dq_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--bins_path "data/dq/Mixed-Python-92k_bins10.jsonl" \
--output_dir "data/train/dq" \
--sample_num 10000
```

#### K-Center

Employ K-Center method for data selection.

```bash
python data_selection/kcenter/kcenter_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--pt_path "data/train/Mixed-Python-92k.pt" \
--output_dir "data/train/kcenter" \
--sample_num 10000
```

#### Random

Select data randomly.

```bash
python data_selection/random/random_select.py \
--data_path "data/train/Mixed-Python-92k.jsonl" \
--output_dir "data/train/random" \
--sample_num 10000
```

</details>

### Training

Finetune the [DeepSeekCoder-Base-6.7B](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) model using the command provided below.

```bash
bash sft/scripts/sft_fsdp.sh deepseek-ai/deepseek-coder-6.7b-base data/train/Mixed-Python-92k-sn10K.jsonl pack 4096 weights/ds-base-7b-sn10K
```

### Evaluation

First, generate code solutions for HumanEval/MBPP benchmarks using the fine-tuned model. All solutions will be saved in the `evalplus_results` directory. Generated text may include non-compilable code or natural language lines and needs to be sanitized for fair comparison. Finally, run the evaluation script to compute the Pass@1 (\%) metric.

We have provided the generated code from other data selection methods in the `evalplus_results` folder for comparison.

```bash
export HUMANEVAL_OVERRIDE_PATH="eval/data/HumanEvalPlus-v0.1.10.jsonl"
export MBPP_OVERRIDE_PATH="eval/data/MbppPlus-v0.2.0.jsonl"

# HumanEval benchmark
python eval/codegen.py \
--dataset humaneval \
--backend vllm \
--greedy \
--model weights/ds-base-7b-sn10K

evalplus.sanitize --samples evalplus_results/humaneval/ours/weights--ds-base-7b-sn10K_vllm_temp0.0.jsonl
evalplus.evaluate --dataset humaneval --samples evalplus_results/humaneval/ours/weights--ds-base-7b-sn10K_vllm_temp0.0-sanitized.jsonl

# MBPP benchmark
python eval/codegen.py \
--dataset mbpp \
--backend vllm \
--greedy \
--model weights/ds-base-7b-sn10K

evalplus.sanitize --samples evalplus_results/mbpp/ours/weights--ds-base-7b-sn10K_vllm_temp0.0.jsonl
evalplus.evaluate --dataset humaneval --samples evalplus_results/mbpp/ours/weights--ds-base-7b-sn10K_vllm_temp0.0-sanitized.jsonl
```
