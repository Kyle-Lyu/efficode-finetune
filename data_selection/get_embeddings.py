import os

#  for debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


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


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_args():
    parser = argparse.ArgumentParser(description="Extract instructions' embeddings")
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the input data file (JSON or JSONL format)."
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to save the results file."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the sentence-transformers model."
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(device).eval()

    # load data
    data = load_json(args.data_path)
    instructions = [item[INST_KEY] for item in data]
    ids = [item[ID_KEY] for item in data]

    id_to_embedding = {}
    with tqdm(desc="Extract instructions' embeddings", total=len(instructions)) as pbar:
        for i in range(0, len(instructions), args.batch_size):
            inst_batch = instructions[i:i+args.batch_size]
            id_batch = ids[i:i + args.batch_size]

            # tokenize sentences
            encoded_input = tokenizer(inst_batch, padding=True, truncation=True, return_tensors='pt').to(device)
            # compute token embeddings
            with torch.no_grad():
                output = model(**encoded_input)
            # perform pooling
            inst_embeddings = mean_pooling(output, encoded_input['attention_mask'])
            # normalize embeddings
            inst_embeddings = F.normalize(inst_embeddings, p=2, dim=1)
            
            for idx, embedding in zip(id_batch, inst_embeddings):
                id_to_embedding[str(idx)] = embedding.cpu() 

            pbar.update(len(inst_batch))
    # save to local using torch.save
    torch.save(id_to_embedding, args.save_path)
    print(f"Embeddings and IDs saved to {args.save_path}")


if __name__ == "__main__":
    main()