import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

QUERY_PFX = "Instruct: Given a description of a grasp, retrieve grasp descriptions that describe similar grasps on similar objects\nQuery: "

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    return parser.parse_args()

def main():
    args = get_args()

    nv_embed = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, model_kwargs={"torch_dtype": "bfloat16"})
    nv_embed.max_seq_length = 32768
    nv_embed.tokenizer.padding_side="right"
    nv_embed.eval()

    def add_eos(texts: list[str]):
        return [text + nv_embed.tokenizer.eos_token for text in texts]

    df = pd.read_csv(args.dataset_path)
    texts = df["text"].tolist()

    # TODO: add torch distributed data parallel
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            query_embeddings = nv_embed.encode(add_eos(texts), batch_size=args.batch_size, show_progress_bar=True, prompt=QUERY_PFX, normalize_embeddings=True)
    np.save(args.output_path, query_embeddings)

if __name__ == "__main__":
    main()
