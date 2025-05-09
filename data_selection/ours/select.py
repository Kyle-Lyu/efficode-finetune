import json
import os
import time
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from utils import *


torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


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


class SampleModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance, slice):
        super(SampleModel, self).__init__()
        #  features.shape: (n, c)
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance
        self.slice = slice
        if slice is None:
            self.slice = self.total_num

        self.init = init
        self.distance = distance

        # centroids.shape: (k, c)
        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()
        print(f"centroids shape: {self.centroids.shape}")

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(get_distance, type=self.distance)
            sample_ids = farthest_distance_sample(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        sample_slice_num = math.ceil(1.0 * self.sample_num / self.slice)
        batch_slice_num = math.ceil(1.0 * self.total_num / self.slice)

        prod_exp_pos = []
        pos_k = []
        for sid in range(batch_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            prod = torch.matmul(self.features[start: end], centroids.transpose(1, 0))  # (slice_num, k)
            prod = prod / self.temperature
            prod_exp = torch.exp(prod)
            prod_exp_pos_cur, pos_k_cur = torch.max(prod_exp, dim=1)  # (slice_num, )
            prod_exp_pos.append(prod_exp_pos_cur)
            pos_k.append(pos_k_cur)
        pos_k = torch.cat(pos_k, dim=0)
        prod_exp_pos = torch.cat(prod_exp_pos, dim=0)

        cent_prob_exp_sum = []
        for sid in range(sample_slice_num):
            start = sid * self.slice
            end = (sid + 1) * self.slice
            cent_prod = torch.matmul(centroids.detach(), centroids[start:end].transpose(1, 0))  # (k, slice_num)
            cent_prod = cent_prod / self.temperature
            cent_prod_exp = torch.exp(cent_prod)
            cent_prob_exp_sum_cur = torch.sum(cent_prod_exp, dim=0)  # (slice_num, )
            cent_prob_exp_sum.append(cent_prob_exp_sum_cur)
        cent_prob_exp_sum = torch.cat(cent_prob_exp_sum, dim=0)

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J


def optimize_dist(features, sample_num, args):
    #  features.shape: (n, c)
    sample_model = SampleModel(features, sample_num, args.temperature, args.init, args.distance, args.balance, args.slice)
    sample_model = sample_model.cuda()

    optimizer = optim.Adam(sample_model.parameters(), lr=args.lr)
    if args.scheduler != "none":
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iter, eta_min=1e-6)
        else:
            raise NotImplementedError

    with tqdm(total=args.max_iter, desc="Training") as pbar:
        for i in range(args.max_iter):
            loss = sample_model.get_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.scheduler != "none":
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_description(f"Iter: {i}, lr: {lr:.6f}, loss: {loss.item():.4f}")
            pbar.update(1)

    # centroids.shape: (k, c)
    centroids = sample_model.centroids.detach()
    centroids = F.normalize(centroids, dim=1)
    slice = 1000
    sample_slice_num = math.ceil(centroids.shape[0] / slice)
    sample_ids = set()
    # _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for sid in range(sample_slice_num):
        start = sid * slice
        end = min((sid + 1) * slice, centroids.shape[0])
        dist = torch.matmul(centroids[start:end], features.transpose(1, 0))  # (slice_num, n)
        _, ids_sort = torch.sort(dist, dim=1, descending=True)
        for i in range(ids_sort.shape[0]):
            for j in range(ids_sort.shape[1]):
                if ids_sort[i, j].item() not in sample_ids:
                    sample_ids.add(ids_sort[i, j].item())
                    break
    print(f"Final count of unique sample IDs: {len(sample_ids)}")
    sample_ids = list(sample_ids)
    return sample_ids


def get_args():
    parser = argparse.ArgumentParser(description="ActiveFT Sampling")
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
    parser.add_argument(
        "--sample_num", type=int, required=True, 
        help="Number of samples to select."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument("--max_iter", type=int, default=300, help="Max iterations.")
    parser.add_argument("--slice", type=int, default=20000, help="Batch size for processing embeddings in chunks to avoid memory overflow.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for softmax.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--init", type=str, default="random", choices=["random", "fps"], help="Initialization method.")
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "cosine"], help="Distance metric.")
    parser.add_argument("--scheduler", type=str, default="none", help="Learning rate scheduler.")
    parser.add_argument("--balance", type=float, default=1.0, help="Balance ratio.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.seed:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    id_to_embedding = torch.load(args.pt_path)
    ids = list(id_to_embedding.keys())
    embeddings = torch.stack([id_to_embedding[id] for id in ids]).cuda()
    # Load data
    data = load_json(args.data_path)
    assert len(data) == len(embeddings), "Mismatch between data and embeddings length."

    print(f"Loaded {len(data)} samples from {args.data_path}")
    print(f"Loaded {len(embeddings)} embeddings from {args.pt_path}")

    total_samples = embeddings.shape[0]
    sample_size = args.sample_num
    print(f"Sampling {sample_size} samples")

    embeddings = F.normalize(embeddings, dim=1)

    start_time = time.time() 
    selected_indices = optimize_dist(embeddings, sample_size, args)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总耗时
    print(f"Sampling completed in {elapsed_time:.2f} seconds.")  # 打印耗时

    selected_indices.sort()

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


if __name__ == '__main__':
    main()
