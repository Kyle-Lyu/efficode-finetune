import os
import json
import argparse
import torch
from tqdm import tqdm


ID_KEY = "id"
LANG_KEY = "language"
INST_KEY = "instruction"
RESP_KEY = "response"
SOURCE_KEY = "source"

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

def format_sample_num(sample_num):
    if sample_num % 1000 == 0:
        return f"{sample_num // 1000}K"
    else:
        return str(sample_num)

def compute_distance(matrix, matrix_2, normalize_emb=True, distance_metric='cosine'):
    """
    Compute distance between two matrices.
    """
    if normalize_emb:
        matrix = matrix / matrix.norm(dim=1, keepdim=True)
        matrix_2 = matrix_2 / matrix_2.norm(dim=1, keepdim=True)

    if distance_metric == 'cosine':
        return torch.mm(matrix, matrix_2.t())
    elif distance_metric == 'manhattan':
        return torch.cdist(matrix, matrix_2, p=1)
    else:
        raise ValueError("Metric not supported. Only support cosine and manhattan")

def distance_chunk_by_chunk(existing_emb, cur_emb, chunk_size, device, distance_metric='cosine'):
    """
    Compute distance between existing embeddings and current embeddings in chunks.
    """
    distance_placeholder = torch.zeros((cur_emb.size(0), existing_emb.shape[0]), dtype=torch.float32).to(device)

    for i in range(0, existing_emb.shape[0], chunk_size):
        chunk_embeddings = existing_emb[i:i + chunk_size].to(device)
        distance_matrix = compute_distance(cur_emb, chunk_embeddings, distance_metric=distance_metric)
        actual_chunk = distance_matrix.size(1)
        distance_placeholder[:, i:i + actual_chunk] = distance_matrix

    return distance_placeholder

def filter_embeddings(embeddings, threshold=0.5, batch_size=10, chunk_size=100000, device='cpu', distance_metric='cosine'):
    """
    Filter embeddings based on distance threshold.
    """
    filtered_indices = [0]  # Start with the first embedding
    existing_emb = embeddings[0].unsqueeze(0).to(device)

    # Initialize the progress bar
    pbar = tqdm(total=len(embeddings) - 1, desc="Filtering embeddings")

    for i in range(1, len(embeddings), batch_size):
        cur_emb = torch.stack(embeddings[i:i + batch_size]).to(device)

        # Compute distance with existing embeddings
        distance_existed = distance_chunk_by_chunk(existing_emb, cur_emb, chunk_size, device, distance_metric)
        distance_existed_bool = torch.any(distance_existed > threshold, dim=1)

        # Compute distance within the current batch
        distance_cur = distance_chunk_by_chunk(cur_emb, cur_emb, chunk_size, device, distance_metric)
        distance_cur = distance_cur.tril(-1)
        distance_cur_bool = torch.any(distance_cur > threshold, dim=1)

        # Combine the two distance checks
        distance_bool = distance_existed_bool | distance_cur_bool

        # Add new embeddings that are not too close to existing ones
        new_indices = [idx for idx, keep in enumerate(distance_bool) if not keep]
        filtered_indices.extend([i + j for j in new_indices])

        # Update existing embeddings
        if new_indices:
            existing_emb = torch.cat([existing_emb, cur_emb[new_indices]], dim=0)

        # Manually update the progress bar
        pbar.update(cur_emb.shape[0])

    # Close the progress bar
    pbar.close()

    return filtered_indices

def get_args():
    parser = argparse.ArgumentParser(description="DEITA Sampling")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--analysis_path", type=str, required=True,
        help="Path to the analysis JSONL file."
    )
    parser.add_argument(
        "--pt_path", type=str, required=True,
        help="Path to the embeddings PT file."
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
        "--threshold", type=float, default=0.9,
        help="Filter threshold (default: 0.9)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100000,
        help="Chunk size for more efficient GPU computing (default: 100000)"
    )
    parser.add_argument(
        "--distance_metric", type=str, default="cosine",
        help="Distance metric (default: cosine)"
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_json(args.data_path)

    # Load analysis data
    analysis_data = load_json(args.analysis_path)
    analysis_data.sort(key=lambda x: x["final_score"], reverse=True)  # Sort by final_score in descending order

    # Load embeddings from .pt file
    id_to_embedding = torch.load(args.pt_path)

    # Sort embeddings based on the order of IDs in analysis_data
    embeddings = []
    ids = []
    for item in analysis_data:
        id_str = str(item[ID_KEY])
        if id_str in id_to_embedding:
            embeddings.append(id_to_embedding[id_str])
            ids.append(id_str)
        else:
            print(f"Warning: ID {id_str} not found in embeddings.")

    # Filter embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filtered_indices = filter_embeddings(
        embeddings, threshold=args.threshold, batch_size=10000, 
        chunk_size=args.chunk_size, device=device, distance_metric=args.distance_metric
    )

    # Split ids into selected and unselected based on filtered_indices
    selected_ids = []
    unselected_ids = []

    for idx, data_id in enumerate(ids):
        if idx in filtered_indices:
            selected_ids.append(data_id)
        else:
            unselected_ids.append(data_id)

    # Get selected IDs
    selected_ids = selected_ids[:args.sample_num]
    if len(selected_ids) < args.sample_num:
        additional_ids = unselected_ids[:args.sample_num - len(selected_ids)]
        selected_ids.extend(additional_ids)

    # Filter data based on selected IDs
    id_to_data = {str(item["id"]): item for item in data}
    sampled_data = [id_to_data[id] for id in selected_ids]

    # Save sampled data
    input_file_name = os.path.basename(args.data_path).replace(".jsonl", "")
    output_file_name = f"{input_file_name}-sn{format_sample_num(args.sample_num)}.jsonl"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    write_jsonl(sampled_data, output_file_path)
    print(f"Sampled data saved to {output_file_path}")

if __name__ == '__main__':
    main()