import argparse
import os
import json
import torch
import numpy as np
import time
from submodular_function import GraphCut
from submodular_optimizer import NaiveGreedy


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


def dataset_quantization(embeddings, k=10):
    """
    Perform dataset quantization using the provided embeddings.
    
    Args:
        embeddings: Numpy array of embeddings corresponding to the data samples.
        k: Number of bins for quantization.
    
    Returns:
        List of bins, where each bin contains the indices of the data samples.
    """
    total_samples = embeddings.shape[0]
    budget_n = total_samples // k
    print(f"Total samples: {total_samples}")
    print(f"Number of bins (k): {k}, Budget per bin: {budget_n}")

    embeddings_original = embeddings.copy()
    indices_original = np.arange(total_samples)
    indices = indices_original.copy()

    sim_matrix = lambda a, b: embeddings[a] @ embeddings[b].T

    # Bin generation
    bins = []
    for i in range(k):
        print(f"Processing bin {i + 1}/{k}")
        submod_f = GraphCut(index=indices, similarity_kernel=sim_matrix)
        submod_opt = NaiveGreedy(args=None, index=indices, budget=budget_n)
        result_indices = submod_opt.select(
            gain_function=submod_f.calc_gain,
            update_state=submod_f.update_state,
        )

        bins.append(result_indices)
        indices = np.delete(indices_original, np.concatenate(bins))
        embeddings = np.delete(embeddings_original, np.concatenate(bins), axis=0)

    print(f"Generated {len(bins)} bins")
    return bins


def get_args():
    parser = argparse.ArgumentParser(description="Dataset Quantization Sampling")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--pt_path", type=str, required=True,
        help="Path to the embeddings PT file."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the sampled file."
    )
    parser.add_argument("--k", type=int, default=10, help="Number of bins for quantization.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    id_to_embedding = torch.load(args.pt_path)
    ids = list(id_to_embedding.keys())
    embeddings = np.array([id_to_embedding[id].cpu().numpy() for id in ids])
    print(f"Loaded {len(embeddings)} embeddings from {args.pt_path}")

    # Load data (only for validation)
    data = load_json(args.data_path)
    print(f"Loaded {len(data)} samples from {args.data_path}")

    assert len(data) == len(embeddings), "Mismatch between data and embeddings length."

    start_time = time.time()
    bins = dataset_quantization(embeddings, k=args.k)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Quantization completed in {elapsed_time:.2f} seconds.")

    # Convert bin indices to ids
    id_bins = []
    for bin_indices in bins:
        id_bin = [ids[i] for i in bin_indices]
        id_bins.append(id_bin)

    # Save bins to a JSONL file
    output_file_name = os.path.basename(args.data_path).replace(".jsonl", f"_bins{args.k}.jsonl")
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(id_bins, output_file_path)
    print(f"Bins saved to {output_file_path}")

    for i, bin_data in enumerate(id_bins):
        print(f"Bin {i + 1} length: {len(bin_data)}")


if __name__ == "__main__":
    main()