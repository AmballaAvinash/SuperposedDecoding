import json
import os
import pickle
from datetime import datetime

# import evaluate
import torch
from tqdm import tqdm

from eval import *
from superposed.llama.metrics import *
from superposed.llama.generation import Llama
from superposed.llama.superposed_generation import SuperposedLlama
from superposed.llama.tokenizer import Tokenizer
from superposed.ngrams.ngram_models import make_models



sup_device = torch.device("cuda:0")
tokenizer = Tokenizer('7B/tokenizer.model')


# Params
param_file = "params/p15_d3_mixed.json"
with open(param_file, "r") as f:
    params = json.load(f)
    print(f"Parameters: {params}")
alpha = params["alpha"]
temp = params["temp"]
n_drafts = params["n_drafts"]
prompt_len = params["prompt_len"]
n_token_sample = params["n_token_sample"]
i_weights = params["i_weights"]
i_length = params["i_length"]



weight_path = "7B/llama-2-7b"
model = SuperposedLlama.build(ckpt_dir=weight_path, 
                         tokenizer_path=f'{weight_path}/tokenizer.model', 
                         max_seq_len=100, 
                         max_batch_size=32,
                         device=sup_device,
                         model_parallel_size=1)



def decode(tokenizer, encoding):
    """
    Args:
        tokenizer (Any): Tokenizer
        encoding (torch.Tensor): Encoding
    Returns:
        decoding (str)
    """
    eos_locs = (encoding == tokenizer.eos_id).nonzero()
    if len(eos_locs > 0):
        encoding = encoding[:eos_locs[0]]
    return tokenizer.decode(encoding.to(torch.int32).tolist())






prompts = [
    "Hi my name is",
    "The Seattle Seahawks were Super Bowl",
    "Penguins are birds native to"
]
tokenized_prompts = tokenizer.encode(prompts, True, False)


alive_gens, _ = model.sup_generate(prompt_tokens=tokenized_prompts, 
                                        smoothing="geom",
                                        max_gen_len=10, 
                                        n_token_sample=n_token_sample,
                                        alpha=alpha, 
                                        temp=temp,
                                        n_drafts=n_drafts,
                                        i_weights=i_weights,
                                        i_length=i_length,
                                        ngrams=ngrams,
                                        get_time=False,
                                        penalty=200)


gens = alive_gens[0].reshape(len(prompts) * n_drafts, -1)
for i in gens:
    print(decode(tokenizer, i))