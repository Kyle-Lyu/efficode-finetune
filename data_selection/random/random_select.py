import json
import argparse
import os
import random


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
    parser = argparse.ArgumentParser(description="Randomly sample data from a JSONL file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--sample_num", type=int, required=True, help="Number of samples to select.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the sampled file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Random seed set to {args.seed} for reproducibility.")

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_json(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    sample_size = args.sample_num
    sampled_data = random.sample(data, sample_size)
    print(f"Sampling {len(sampled_data)} samples")

    input_file_name = os.path.basename(args.data_path).replace(".jsonl", "")
    output_file_name = f"{input_file_name}-sn{format_sample_num(args.sample_num)}.jsonl"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(sampled_data, output_file_path)
    print(f"Sampled data saved to {output_file_path}")

if __name__ == "__main__":
    main()