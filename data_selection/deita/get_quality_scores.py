import os

#  for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import json
import time
import argparse
from tqdm import tqdm
from selection.scorer import Llama_Scorer


ID_KEY = "id"
LANG_KEY = "language"
INST_KEY = "instruction"
RESP_KEY = "response"
SOURCE_KEY = "source"


def load_json(file_path):
    try:
        data = json.load(open(file_path, "r"))
    except json.JSONDecodeError:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    
    return data


def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii = False) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Calculate quality scores for instruction-response pairs.")
    
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input data file (JSON or JSONL format)."
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to save the results file (JSONL format)."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the pre-trained quality scorer model."
    )
    parser.add_argument(
        "--is_vllm", type=bool, default=True,
        help="Whether to use vLLM for inference. Default is True."
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for vLLM. Default is 0.9."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    # load quality scorer model
    scorer = Llama_Scorer(args.model_path, is_vllm=args.is_vllm, gpu_memory_utilization=args.gpu_memory_utilization)
    
    # load data
    data = load_json(args.data_path)

    # calculate quality scores
    results = []
    start_time = time.time()
    for item in tqdm(data, desc=f"Processing on device {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"):
        inst = item[INST_KEY]
        resp = item[RESP_KEY]

        quality_score = scorer.infer_quality(inst, resp)

        result = {
            ID_KEY: item[ID_KEY],
            SOURCE_KEY: item[SOURCE_KEY],
            "quality_score": quality_score,
        }
        results.append(result)
    end_time = time.time()
    time_used = end_time - start_time

    # save results to file
    write_jsonl(results, args.save_path)

    print(
        "\nProcessing completed!\n"
        f"  - Time used: {time_used:.2f} seconds\n"
        f"  - Results saved to: {args.save_path}"
    )

if __name__ == "__main__":
    main()