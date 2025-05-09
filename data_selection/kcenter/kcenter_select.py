import os
import time
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


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


def k_center_greedy(features: np.ndarray, num_samples: int, metric: str = "euclidean") -> np.ndarray:
    """
    K-Center Greedy sampling algorithm.

    Args:
        features: Embedding matrix of shape (n_samples, n_features).
        num_samples: Number of samples to select.
        metric: Distance metric (default: euclidean).

    Returns:
        Selected indices (shape: [num_samples]).
    """
    min_distances = None
    selected_indices = []

    # Initialize with random first point
    first_idx = np.random.choice(np.arange(len(features)))
    selected_indices.append(first_idx)

    # Update distances for the first center
    min_distances = pairwise_distances(features, features[first_idx].reshape(1, -1), metric=metric)

    for _ in tqdm(range(1, num_samples), desc="K-Center Greedy Sampling"):
        # Find the point with maximum minimum distance
        new_idx = np.argmax(min_distances)
        assert new_idx not in selected_indices

        selected_indices.append(new_idx)

        # Update distances with the new center
        new_distances = pairwise_distances(features, features[new_idx].reshape(1, -1), metric=metric)
        min_distances = np.minimum(min_distances, new_distances)

    return np.array(selected_indices)


def get_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="K-Center Greedy Sampling")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--pt_path", type=str, required=True,
        help="Path to .pt file containing embeddings."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the sampled file."
    )
    parser.add_argument(
        "--sample_num", type=int, required=True, 
        help="Number of samples to select."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.seed:
        np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    id_to_embedding = torch.load(args.pt_path) 
    ids = list(id_to_embedding.keys())
    embeddings = np.array([id_to_embedding[id].cpu().numpy() for id in ids])

    # Load data
    data = load_json(args.data_path)
    assert len(data) == len(embeddings), "Mismatch between data and embeddings length."

    print(f"Loaded {len(data)} samples from {args.data_path}")
    print(f"Loaded {len(embeddings)} embeddings from {args.pt_path}")

    # Calculate sample count
    total_samples = len(embeddings)
    sample_size = args.sample_num
    print(f"Sampling {sample_size} samples")

    # Run K-Center Greedy
    start_time = time.time()
    selected_indices = k_center_greedy(embeddings, sample_size)
    end_time = time.time()
    print(f"K-Center Greedy sampling completed in {end_time - start_time:.2f} seconds.")

    # Get selected IDs
    selected_ids = [ids[i] for i in selected_indices]

    # Filter data based on selected IDs
    id_to_data = {str(item["id"]): item for item in data}
    sampled_data = [id_to_data[id] for id in selected_ids]

    # Save sampled data
    input_file_name = os.path.basename(args.data_path).replace(".jsonl", "")
    output_file_name = f"{input_file_name}-sn{format_sample_num(args.sample_num)}.jsonl"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(sampled_data, output_file_path)
    print(f"Sampled data saved to {output_file_path}")


if __name__ == "__main__":
    main()