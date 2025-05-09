import argparse
import os
import json
import time
import random


def format_sample_num(sample_num):
    if sample_num % 1000 == 0:
        return f"{sample_num // 1000}K"
    else:
        return str(sample_num)


def load_json(file_path):
    """
    Load JSON or JSONL file.
    """
    try:
        data = json.load(open(file_path, "r"))
    except json.JSONDecodeError:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    """
    Save data to a JSONL file.
    """
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="Dataset Quantization Sampling")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input JSONL file containing the full dataset."
    )
    parser.add_argument(
        "--bins_path", type=str, required=True,
        help="Path to the JSONL file containing the bins (each line is a bin)."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the sampled file."
    )
    parser.add_argument(
        "--sample_num", type=int, required=True, 
        help="Total number of samples to select."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    return args


def sample_from_bins(bins, data, total_sample_num, seed):
    """
    Randomly sample from each bin, ensuring the total number of samples is equal to total_sample_num.
    
    Args:
        bins: List of bins, where each bin is a list of IDs.
        data: Full dataset (list of dictionaries).
        total_sample_num: Total number of samples to select.
        seed: Random seed for reproducibility.
    
    Returns:
        List of sampled data items.
    """
    random.seed(seed)
    sampled_data = []
    id_to_data = {str(item["id"]): item for item in data}  # Create a mapping from ID to data item

    # Calculate the number of samples per bin
    num_bins = len(bins)
    samples_per_bin = total_sample_num // num_bins
    remaining_samples = total_sample_num % num_bins  # Handle the remainder

    for i, bin_indices in enumerate(bins):
        bin_data = [id_to_data[id] for id in bin_indices if id in id_to_data]
        current_bin_sample_num = samples_per_bin + (1 if i < remaining_samples else 0)

        if len(bin_data) > current_bin_sample_num:
            sampled_bin = random.sample(bin_data, current_bin_sample_num)
        else:
            sampled_bin = bin_data  # If the bin has fewer items than sample_num, take all
        sampled_data.extend(sampled_bin)

    return sampled_data


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load bins
    bins = load_json(args.bins_path)
    print(f"Loaded {len(bins)} bins from {args.bins_path}")

    # Load full dataset
    data = load_json(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    # Sample from bins
    start_time = time.time()
    sampled_data = sample_from_bins(bins, data, args.sample_num, args.seed)
    end_time = time.time()
    print(f"Sampled {len(sampled_data)} data in total.")
    sampling_time = end_time - start_time
    print(f"Sampling completed in {sampling_time:.2f} seconds.")

    # Save sampled data
    input_file_name = os.path.basename(args.data_path).replace(".jsonl", "")
    output_file_name = f"{input_file_name}-sn{format_sample_num(args.sample_num)}.jsonl"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(sampled_data, output_file_path)
    print(f"Sampled data saved to {output_file_path}")


if __name__ == "__main__":
    main()
