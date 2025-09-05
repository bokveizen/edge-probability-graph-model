from collections import Counter
from itertools import combinations
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
from tqdm.auto import tqdm, trange

dataset_list = [
    "facebook",
    "hamsterster",
    "web-spam",
    "polblogs",
    "bio-CE-PG",
    "bio-SC-HT",
]

p_data = Path(".")
p_nx = p_data / "nx_graph"

p_sbm = p_data / "orig_sbm"
p_sbm.mkdir(exist_ok=True)

p_sbm_pn = p_data / "sbm_PB_NB"

for ds in dataset_list:
    NB = np.load(p_sbm_pn / f"{ds}_N_blocks.npy")
    PB = np.load(p_sbm_pn / f"{ds}_p_blocks.npy")

    # expand each entry in PB to a block
    # PB[i, j] expand to a block of size NB[i] * NB[j]
    p_full = np.zeros((NB.sum(), NB.sum()))
    for i in range(PB.shape[0]):
        for j in range(PB.shape[1]):
            block = np.full((NB[i], NB[j]), PB[i, j])
            p_full[
                np.sum(NB[:i]) : np.sum(NB[: i + 1]),
                np.sum(NB[:j]) : np.sum(NB[: j + 1]),
            ] = block

    N = NB.sum()
    uv2p = dict()
    for u, v in combinations(range(N), 2):
        uv2p[(u, v)] = p_full[u, v]
    uv_array = np.array(list(uv2p.keys()))
    p_array = np.array(list(uv2p.values()))

    # generate graphs
    for i_graph in range(100):
        # edge pair is sampled with probability p
        sampled_indices = np.random.random(p_full.shape) < p_full
        sampled_uv = uv_array[sampled_indices]
        # save the sampled graph
        with open(p_sbm / f"{ds}_{i_graph}.txt", "w") as f:
            # f.write(f"{N} {sampled_uv.shape[0]}\n")
            for u, v in sampled_uv:
                u, v = min(u, v), max(u, v)
                f.write(f"{u} {v}\n")
