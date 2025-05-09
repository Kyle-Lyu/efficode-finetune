import os

#  for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["HUMANEVAL_OVERRIDE_PATH"] = "eval/data/HumanEvalPlus-v0.1.10.jsonl"
# os.environ["MBPP_OVERRIDE_PATH"] = "eval/data/MbppPlus-v0.2.0.jsonl"

import json
import argparse
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from utils import INST_PREFIX, RESP_PREFIX, make_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, choices=["humaneval", "mbpp"], required=True,
        help="The dataset to use for evaluation. Choose from `humaneval` or `mbpp`."
    )
    parser.add_argument(
        "--samples_dir", type=str, default="evalplus_results",
        help="Directory to save generated samples. Default is `evalplus_results`."
    )
    parser.add_argument(
        "--backend", type=str, choices=["hf", "vllm"], default="vllm",
        help="The backend to use for model inference. Choose from `hf` (HuggingFace) or `vllm` (vLLM). Default is `vllm`.",
    )
    parser.add_argument(
        "--model", type=str, required=True, 
        help="Path to the model. This is a required argument."
    )
    parser.add_argument(
        "--base", action="store_true", 
        help="Whether the model is a base model. If set, it indicates a base model."
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16",
        help="Data type for model inference. Choose from `float16` or `bfloat16`. Default is `bfloat16`."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, 
        help="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt. Default is 1024."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, 
        help="The value used to modulate the next token probabilities. Default is 0.7."
    )
    parser.add_argument(
        "--n_samples", type=int, default=1,
        help="Number of samples to generate for each task. Default is 1."
    )
    parser.add_argument(
        "--greedy", action="store_true", 
        help="Whether to use greedy decoding strategy. If set, greedy decoding will be used."
    )
    #  for vllm 
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache."
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="The number of GPUs to use for distributed execution with tensor parallelism."
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # use greedy decoding strategy
    if args.greedy and (args.temperature != 0 or args.n_samples != 1):
        args.temperature = 0.0
        args.n_samples = 1
    
    print("Hyperparameters as follows:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # make project dir
    os.makedirs(args.samples_dir, exist_ok=True)
    # make dataset dir
    os.makedirs(os.path.join(args.samples_dir, args.dataset), exist_ok=True)

    # local path for saving generated codes
    identifier = args.model.strip("./").replace("/", "--") + f"_{args.backend}_temp{args.temperature}"
    target_path = os.path.join(args.samples_dir, args.dataset, f"{identifier}.jsonl")
    print(f"Code outputs will be saved to {target_path}")

    # Model instructions
    instruction_prefix = INST_PREFIX
    response_prefix = RESP_PREFIX

    # load model runner
    model_runner = make_model(
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        batch_size=args.n_samples,
        temperature=args.temperature,
        force_base_prompt=args.base,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # load dataset
    if args.dataset == "humaneval":
        dataset = get_human_eval_plus()
    elif args.dataset == "mbpp":
        dataset = get_mbpp_plus()
    else:
        raise ValueError(f"Supported datasets are `humaneval` and `mbpp`, but you provided {args.dataset}")

    task2nexist = {}
    if os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1
    
    # generate code samples
    for task_id, task in tqdm(dataset.items(), desc=f"{args.dataset} benchmark"):
        n_more_samples = args.n_samples
        if task2nexist.get(task_id, 0) > 0:
            n_more_samples -= task2nexist[task_id]

        task_prompt = task["prompt"].strip() + "\n"
        outputs = model_runner.codegen(
            prompt=task_prompt,
            do_sample=not args.greedy,
            num_samples=n_more_samples,
        )
        assert outputs, f"No outputs from model when handling {task_id}!"

        # save to local
        for impl in outputs:
            solution = task_prompt + impl if model_runner.is_direct_completion() else impl
            with open(target_path, "a") as f:
                f.write(
                    json.dumps(
                        {"task_id": task_id, "solution": solution}
                    ) + "\n"
                )


if __name__ == "__main__":
    main()
