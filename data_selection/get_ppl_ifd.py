import os

#  for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import json
import math
import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm


CHAT_TEMPLATE = '''{% if not add_generation_prompt is defined %}
{% set add_generation_prompt = false %}
{% endif %}
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {%- set ns.found = true -%}
    {%- endif -%}
{%- endfor -%}
{{bos_token}}{%- if not ns.found -%}
{{'You are a helpful programming assistant.\n'}}
{%- endif %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{ message['content'] }}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'### Instruction:\n' + message['content'] + '\n'}}
        {%- else %}
{{'### Response:\n' + message['content']}}{{eos_token}}{{'\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}
{{'### Response:'}}
{% endif %}'''

SEPARATOR = "### Response:\n"
ID_KEY = "id"
LANG_KEY = "language"
INST_KEY = "instruction"
RESP_KEY = "response"
SOURCE_KEY = "source"
IGNORE_TOKEN_ID = -100


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


def calculate_ppl_ifd(
    tokenizer: transformers.PreTrainedTokenizerBase, 
    model: transformers.PreTrainedModel, 
    instruction: str, response: str, max_length: int
):
    messages = [
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    end_index = len(tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_length))
    response = response + tokenizer.eos_token

    # perplexity of directly generating response
    try:
        resp_ids = tokenizer.encode(response, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length - end_index + 1).to(model.device)
        with torch.no_grad():
            outputs = model(resp_ids, labels=resp_ids.contiguous())
        loss = outputs.loss
        perplexity_resp = torch.exp(loss).to('cpu').item()
    except Exception as e:
        perplexity_resp = 0

    # perplexity of generating response given instruction
    try:
        full_prompt = prompt + response
        input_ids = tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length).to(model.device)
        labels = input_ids.clone()
        labels[0, :end_index] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        perplexity_text = torch.exp(loss).to('cpu').item()
    except Exception as e:
        perplexity_text = 0

    # calculate IFD score
    try:
        ifd_score = perplexity_text / perplexity_resp
    except ZeroDivisionError:
        ifd_score = 0

    if math.isnan(ifd_score):
        ifd_score = 0

    return perplexity_text, ifd_score


def get_args():
    parser = argparse.ArgumentParser(description="Calculate PPL and IFD scores")
    
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
        help="Path to the pre-trained language model."
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Maximum sequence length for tokenization. If not provided, it will be inferred from the model config."
    )
    parser.add_argument(
        "--start_idx", type=int, default=None,
        help="Start index of the data to process. If not provided, processing starts from the beginning."
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="End index of the data to process. If not provided, processing continues until the end of the data."
    )
    
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    
    # load model and tokenizer
    config = AutoConfig.from_pretrained(args.model_path)
    if args.max_length is None:
        args.max_length = getattr(config, "max_position_embeddings", None)
    if args.max_length is None:
        raise ValueError("Please specify the maximum sample length, for example --max_length 2048")
    
    # print("Hyperparameters as follows:")
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = CHAT_TEMPLATE
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # load data
    data = load_json(args.data_path)
    start_idx = args.start_idx if args.start_idx else 0
    end_idx = args.end_idx if args.end_idx else len(data)
    sampled_data = data[start_idx:end_idx]

    # calculate PPL and IFD scores
    results = []
    start_time = time.time()
    for item in tqdm(sampled_data, desc=f"Process on device {os.environ.get('CUDA_VISIBLE_DEVICES', None)}"):
        instruction = item[INST_KEY]
        response = item[RESP_KEY]

        # calculate IFD score
        ppl, ifd = calculate_ppl_ifd(tokenizer, model, instruction, response, args.max_length)

        # save results
        result = {
            ID_KEY: item[ID_KEY],
            SOURCE_KEY: item[SOURCE_KEY],
            "ppl": ppl,
            "ifd": ifd,
        }
        results.append(result)
    end_time = time.time()
    time_used = end_time - start_time

    # save results to file
    write_jsonl(results, args.save_path)

    print(
        "\nProcessing completed!\n"
        f"  - Time used: {time_used:.2f} seconds\n"
        f"  - Processed indices: {start_idx} - {end_idx}\n"
        f"  - Results saved to: {args.save_path}"
    )


if __name__ == "__main__":
    main()
