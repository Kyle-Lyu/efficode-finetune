import json
import argparse
import os

# 定义关键词
ID_KEY = "id"
LANG_KEY = "language"
INST_KEY = "instruction"
RESP_KEY = "response"
SOURCE_KEY = "source"
PPL_KEY = "ppl"
IFD_KEY = "ifd"

def format_sample_num(sample_num):
    if sample_num % 1000 == 0:
        return f"{sample_num // 1000}K"
    else:
        return str(sample_num)

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
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Sample data by IFD from a JSONL file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--score_path", type=str, required=True, help="Path to the analysis JSONL file.")
    parser.add_argument("--sample_num", type=int, required=True, help="Number of samples to select.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the sampled file.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集和 IFD 分数数据
    data = load_json(args.data_path)
    scores = load_json(args.score_path)

    print(f"Loaded {len(data)} samples from {args.data_path}")
    print(f"Loaded {len(scores)} IFD scores from {args.score_path}")

    # 将 IFD 分数数据转换为字典，方便根据 id 查找
    score_dict = {item[ID_KEY]: item for item in scores}

    # 合并数据和 IFD 分数
    merged_data = []
    for item in data:
        item_id = item[ID_KEY]
        if item_id in score_dict:
            item[IFD_KEY] = score_dict[item_id][IFD_KEY]
            merged_data.append(item)

    # 根据 IFD 分数从大到小排序
    sorted_data = sorted(merged_data, key=lambda x: x[IFD_KEY], reverse=True)

    # 采样数据
    sample_size = args.sample_num
    sampled_data = sorted_data[:sample_size]
    print(f"Sampling {len(sampled_data)} samples")

    # 保存采样后的数据
    input_file_name = os.path.basename(args.data_path).replace(".jsonl", "")
    output_file_name = f"{input_file_name}-sn{format_sample_num(args.sample_num)}.jsonl"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(sampled_data, output_file_path)
    print(f"Sampled data saved to {output_file_path}")

if __name__ == "__main__":
    main()