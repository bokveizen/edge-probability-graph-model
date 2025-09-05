import numpy as np
from itertools import combinations
from tqdm import tqdm

k_dict = dict()
seed_matrix_dict = dict()
seed_matrix_normed_dict = dict()

n_dict = {
    "facebook": 4039,
    "hamsterster": 2000,
    "polblogs": 1222,
    "web-spam": 4767,
    "bio-CE-PG": 1692,
    "bio-SC-HT": 2077,
}
m_dict = {
    "facebook": 88234,
    "hamsterster": 16097,
    "polblogs": 16717,
    "web-spam": 37375,
    "bio-CE-PG": 47309,
    "bio-SC-HT": 63023,
}
seed_dict = {
    "facebook": [0.9999, 0.696477, 0.696417, 0.340615],
    "hamsterster": [0.9999, 0.685853, 0.685843, 0.20854],
    "polblogs": [0.9999, 0.707334, 0.707345, 0.146953],
    "web-spam": [0.9999, 0.614892, 0.614885, 0.134607],
    "bio-CE-PG": [0.9999, 0.806698, 0.806671, 0.206475],
    "bio-SC-HT": [0.9999, 0.70475, 0.7042, 0.227281],
}
ds_list = list(seed_dict.keys())
k_dict = dict()

for ds in ds_list:
    seed_matrix_dict[ds] = np.array(seed_dict[ds]).reshape(2, 2)
    seed_matrix_dict[ds] = (seed_matrix_dict[ds] + seed_matrix_dict[ds].T) / 2
    n = n_dict[ds]
    m = m_dict[ds]
    # k is the smallest integer such that 2^k >= n
    k = np.ceil(np.log2(n)).astype(int)    
    k_dict[ds] = k

if __name__ == "__main__":
    import os    

    os.makedirs("kron_probs", exist_ok=True)
    os.makedirs("orig_kr", exist_ok=True)
    
    datasets = list(seed_matrix_dict.keys())
    for ds in tqdm(datasets):        
        # compute the Kronecker power and save the edge probabilities
        k = k_dict[ds]
        seed_matrix = seed_matrix_dict[ds]
        for i in range(k - 1):
            seed_matrix = np.kron(seed_matrix, seed_matrix_dict[ds])
        N = seed_matrix.shape[0]
        uv2p = dict()
        with open(f"kron_probs/{ds}_kron.txt", "w") as f:
            f.write(f"{N} {N * (N - 1) // 2}\n")
            for u, v in combinations(range(N), 2):
                f.write(f"{u} {v} {seed_matrix[u, v]}\n")
                uv2p[(u, v)] = seed_matrix[u, v]
        uv_array = np.array(list(uv2p.keys()))
        p_array = np.array(list(uv2p.values()))
        # generate graphs
        for i_graph in range(100):
            # edge pair is sampled with probability p
            sampled_indices = np.random.random(p_array.shape) < p_array
            sampled_uv = uv_array[sampled_indices]
            # save the sampled graph
            with open(f"orig_kr/{ds}_{i_graph}.txt", "w") as f:
                # f.write(f"{N} {sampled_uv.shape[0]}\n")
                for u, v in sampled_uv:
                    u, v = min(u, v), max(u, v)
                    f.write(f"{u} {v}\n")
        
