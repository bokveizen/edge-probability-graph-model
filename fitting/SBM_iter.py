import os
import numpy as np
import torch
import argparse
import pickle
import networkx as nx
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# read the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--nround", type=int, default=100_000, help="number of rounds")
parser.add_argument("--alpha", type=float, default=0.001, help="initial alpha")
parser.add_argument("--gpu", type=int, default=0, help="gpu id")
parser.add_argument("--dataset", type=str, default="cora", help="dataset name")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--ep", type=int, default=1000, help="number of epochs")
parser.add_argument("--wt", type=float, default=0.0, help="weight of triangles")
parser.add_argument("--ww", type=float, default=0.0, help="weight of wedges")
parser.add_argument("--wr", type=float, default=0.0, help="weight of ratio")
# parser.add_argument("--wn", type=float, default=0.0, help="weight of nodes")  # not available for iterative binding
parser.add_argument("--name", type=str, default="", help="output folder name")
parser.add_argument("--float64", action="store_true", help="use float64 instead of float32")
args = parser.parse_args()

gpu_id = args.gpu
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

if args.float64:
    torch.set_default_dtype(torch.float64)
else:
    torch.set_default_dtype(torch.float32)
torch.set_num_threads(5)

# independent non-uniform node sampling
# optimize degrees too
# the grouping is kept
# nodes with the same degree are sampled with the same probability (i.e., the same alpha)

num_eps = 1e-8
class iterBindingSBM(torch.nn.Module):
    def __init__(self, alpha: torch.tensor, n_round: int, p_blocks: torch.tensor, N_blocks: torch.tensor):
        # alpha: (n_blocks, ); the binding strength of the nodes in each block
        # n_round: number of rounds
        # p_blocks: (n_blocks, n_blocks); p_blocks[i, j] is the probability of an edge between a node in block i and a node in block j
        # N_blocks: (n_blocks, ); N_blocks[i] is the number of nodes in block i
        
        super().__init__()
        # make alpha a tensor of parameters
        self.alpha = torch.nn.Parameter(alpha.to(device))
        self.n_round = torch.tensor(float(n_round)).to(device)
        self.p_blocks = p_blocks.to(device)
        self.N_blocks = N_blocks.to(device)
        
        assert self.alpha.shape[0] == self.p_blocks.shape[0] == self.p_blocks.shape[1] == self.N_blocks.shape[0], "the number of blocks is not consistent"
        self.n_blocks = self.alpha.shape[0]
    
    def forward(self):
        alpha_sigmoid = torch.sigmoid(self.alpha)        
        P = self.p_blocks
        NB = self.N_blocks
        n_blocks = self.n_blocks
        n_round = self.n_round
        
        i, j, k = torch.meshgrid(torch.arange(n_blocks), torch.arange(n_blocks), torch.arange(n_blocks))
        
        # P_stack: (n_blocks, n_blocks, n_blocks, 3); P_stack[i, j, k] = [P[i, j], P[i, k], P[j, k]]
        P_stack = torch.stack((P[i, j], P[i, k], P[j, k]), dim=-1)
        P_sorted, indices = torch.sort(P_stack, dim=-1)
        P_min, P_min_indices = P_sorted[..., 0], indices[..., 0]
        P_mid, P_mid_indices = P_sorted[..., 1], indices[..., 1]
        P_max, P_max_indices = P_sorted[..., 2], indices[..., 2]
                
        alpha_sigmoid_stack = torch.stack((alpha_sigmoid[k], alpha_sigmoid[j], alpha_sigmoid[i]), dim=-1)
        alpha_max = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_max_indices.unsqueeze(-1)).squeeze(-1)
        alpha_mid = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_mid_indices.unsqueeze(-1)).squeeze(-1)
        alpha_min = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_min_indices.unsqueeze(-1)).squeeze(-1)
        
        # pg_3: (n_deg, n_deg, n_deg); pg_3[i, j, k] = alpha_sigmoid[i] * alpha_sigmoid[j] * alpha_sigmoid[k]        
        pg_3 = alpha_max * alpha_mid * alpha_min
        # pg_2_xxx: (n_deg, n_deg, n_deg)
        # pg_2_max_mid[i, j, k] is the probability that the two endpoints of R_min[i, j, k] are sampled       
        # pg_2_max_min[i, j, k] is the probability that the two endpoints of R_mid[i, j, k] are sampled
        # pg_2_mid_min[i, j, k] is the probability that the two endpoints of R_max[i, j, k] are sampled
        pg_2_min = alpha_max * alpha_mid * (1 - alpha_min)  # corresponding to R_min
        pg_2_mid = alpha_max * (1 - alpha_mid) * alpha_min  # corresponding to R_mid
        pg_2_max = (1 - alpha_max) * alpha_mid * alpha_min  # corresponding to R_max
        
        pg_lt2 = 1 - pg_3 - pg_2_min - pg_2_mid - pg_2_max      
                
        # conditioned on all three pairs are sampled, compute the probability that the motif is sampled
        pg_geq2 = pg_3 + pg_2_min + pg_2_mid + pg_2_max + num_eps
        pg_geq2_ex_max = pg_3 + pg_2_min + pg_2_mid + num_eps
        pg_geq2_ex_mid = pg_3 + pg_2_min + pg_2_max + num_eps
        pg_geq2_ex_min = pg_3 + pg_2_mid + pg_2_max + num_eps
        
        # partition prbabilities: 5 sub-cases
        # pp_3 = pg_3 / pg_geq2
        # pp_max_all = (pg_2_max / pg_geq2) * (pg_3 / (pg_geq2_ex_max))
        # pp_mid_all = (pg_2_mid / pg_geq2) * (pg_3 / (pg_geq2_ex_mid))
        # pp_min_all = (pg_2_min / pg_geq2) * (pg_3 / (pg_geq2_ex_min))
        # pp_indep = 1 - pp_3 - pp_max_all - pp_mid_all - pp_min_all
                
        pp_3 = pg_3 * (1 - pg_lt2 ** (n_round - 1)) / pg_geq2  # when n_round approaches infinity, pp_3 = pg_3 / pg_geq2
        pp_max_all = (pg_3 * (1 - pg_lt2 ** (n_round - 1)) / pg_geq2_ex_max) - pp_3
        pp_mid_all = (pg_3 * (1 - pg_lt2 ** (n_round - 1)) / pg_geq2_ex_mid) - pp_3
        pp_min_all = (pg_3 * (1 - pg_lt2 ** (n_round - 1)) / pg_geq2_ex_min) - pp_3
        pp_indep = 1 - pp_3 - pp_max_all - pp_mid_all - pp_min_all
                
        # final motif probabilities
        p_3 = (
            pp_3 * P_min +
            pp_max_all * P_max * P_min +
            pp_mid_all * P_mid * P_min +
            pp_mid_all * P_mid * P_min +
            pp_indep * P_max * P_mid * P_min
        )
        p_2_max_mid = (
            pp_3 * (P_mid - P_min) +
            pp_max_all * P_max * (P_mid - P_min) +
            pp_mid_all * P_mid * (P_max - P_min) +
            pp_min_all * (1 - P_min) * P_mid + 
            pp_indep * P_max * P_mid * (1 - P_min)
        )
        p_2_max_min = (
            # pp_3 * 0 +
            # pp_max_all * 0 +
            pp_mid_all * (1 - P_mid) * P_min +
            pp_min_all * P_min * (P_max - P_mid) + 
            pp_indep * P_max * (1 - P_mid) * P_min
        )
        p_2_mid_min = (
            # pp_3 * 0 +
            pp_max_all * (1 - P_max) * P_min +
            # pp_mid_all * 0 +
            # pp_min_all * 0 + 
            pp_indep * (1 - P_max) * P_mid * P_min
        )
        p_1_max = (
            pp_3 * (P_max - P_mid) +
            pp_max_all * P_max * (1 - P_mid) +
            pp_mid_all * (1 - P_mid) * (P_max - P_min) +
            pp_min_all * (1 - P_min) * (P_max - P_mid) +
            pp_indep * P_max * (1 - P_mid) * (1 - P_min)
        )
        p_1_mid = (
            # pp_3 * 0 +
            pp_max_all * (1 - P_max) * (P_mid - P_min) +
            pp_mid_all * (1 - P_mid) * (1 - P_max) +
            # pp_min_all * 0 +
            pp_indep * (1 - P_max) * P_mid * (1 - P_min)
        )
        p_1_min = (
            # pp_3 * 0 +
            # pp_max_all * 0 +
            # pp_mid_all * 0 +
            pp_min_all * P_min * (1 - P_max) +
            pp_indep * (1 - P_max) * (1 - P_mid) * P_min
        )
        p_0 = (
            pp_3 * (1 - P_max) +
            pp_max_all * (1 - P_max) * (1 - P_mid) +
            pp_mid_all * (1 - P_mid) * (1 - P_max) +
            pp_min_all * (1 - P_min) * (1 - P_max) +
            pp_indep * (1 - P_max) * (1 - P_mid) * (1 - P_min)
        )
        p_2 = p_2_max_mid + p_2_max_min + p_2_mid_min
        p_1 = p_1_max + p_1_mid + p_1_min
        
        # compute the number of triplets
        # n_triplets: (n_deg, n_deg, n_deg)
        # if i, j, and k are all distinct, n_triplets[i, j, k] = ND[i] * ND[j] * ND[k] / 6 
        # if i == j != k, n_triplets[i, j, k] = ND[i] * (ND[i] - 1) * ND[k] / 6
        # if i == k != j, n_triplets[i, j, k] = ND[i] * (ND[i] - 1) * ND[j] / 6
        # if j == k != i, n_triplets[i, j, k] = ND[j] * (ND[j] - 1) * ND[i] / 6
        # if i == j == k, n_triplets[i, j, k] = ND[i] * (ND[i] - 1) * (ND[i] - 2) / 6
        
        # i, j, k = torch.meshgrid(torch.arange(n_deg), torch.arange(n_deg), torch.arange(n_deg))

        # Calculate n_triplets based on the conditions
        n_triplets = torch.zeros((n_blocks, n_blocks, n_blocks)).to(device)

        # Condition: i, j, and k are all distinct
        mask = (i != j) & (j != k) & (i != k)
        n_triplets[mask] = NB[i[mask]] * NB[j[mask]] * NB[k[mask]] / 6

        # Condition: i == j != k
        mask = (i == j) & (i != k)
        n_triplets[mask] = NB[i[mask]] * (NB[i[mask]] - 1) * NB[k[mask]] / 6

        # Condition: i == k != j
        mask = (i == k) & (i != j)
        n_triplets[mask] = NB[i[mask]] * (NB[i[mask]] - 1) * NB[j[mask]] / 6

        # Condition: j == k != i
        mask = (j == k) & (j != i)
        n_triplets[mask] = NB[j[mask]] * (NB[j[mask]] - 1) * NB[i[mask]] / 6

        # Condition: i == j == k
        mask = (i == j) & (j == k)
        n_triplets[mask] = NB[i[mask]] * (NB[i[mask]] - 1) * (NB[i[mask]] - 2) / 6
        
        # compute the expected number of 3-motifs
        E_0 = (p_0 * n_triplets).sum()
        E_1 = (p_1 * n_triplets).sum()
        E_2 = (p_2 * n_triplets).sum()
        E_3 = (p_3 * n_triplets).sum()
        
        return E_0, E_1, E_2, E_3

p_data = Path("../data/nx_graph")
p_SBM_PBSB = Path("../data/sbm_PB_NB")

with open(p_data / f"{args.dataset}.graph", "rb") as f:
    graph: nx.Graph = pickle.load(f)

# inverse sigmoid
def inv_sigmoid(x: float):
    return torch.log(x / (1 - x))

deg_list = [len(graph[v]) for v in graph]
n_triangles = sum(nx.triangles(graph).values())
n_wedges = sum(d * (d - 1) / 2 for d in deg_list)
n_open = n_wedges - n_triangles
n_triangles /= 3
ratio_gt = n_triangles / n_open
p_blocks = np.load(p_SBM_PBSB / f"{args.dataset}_p_blocks.npy")
N_blocks = np.load(p_SBM_PBSB / f"{args.dataset}_N_blocks.npy")
n_blocks = p_blocks.shape[0]

p_blocks = torch.tensor(p_blocks).to(device)
N_blocks = torch.tensor(N_blocks).to(device)

alpha_raw = [args.alpha for _ in range(n_blocks)]
alpha_inv_sigmoid = inv_sigmoid(torch.tensor(alpha_raw))

model = iterBindingSBM(alpha=alpha_inv_sigmoid, n_round=args.nround, p_blocks=p_blocks, N_blocks=N_blocks).to(device)

# optimize the model (i.e., alpha) using Adam
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epsilon = 1e-3
loss_eps = 1e-16
for i in tqdm(range(args.ep), desc="Training", unit="epoch"):
    optimizer.zero_grad()
    E_0, E_1, E_2, E_3 = model()    
    ratio_gen = E_3 / E_2    
    loss_triangle = (1 - (E_3 / n_triangles)) ** 2
    loss_wedge = (1 - (E_2 / n_open)) ** 2
    loss_ratio = (1 - (ratio_gen / ratio_gt)) ** 2
    # loss_node = (1 - (E_n / n_gt)) ** 2
    loss = args.wt * loss_triangle + args.ww * loss_wedge + args.wr * loss_ratio
    if loss < loss_eps:
        break
    loss.backward()
    # grad climping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)
    # replace nan in gradients with 0
    for p in model.parameters():
        if torch.isnan(p.grad).any():
            p.grad = torch.where(torch.isnan(p.grad), torch.zeros_like(p.grad), p.grad)
    optimizer.step()
    with torch.no_grad():
        # model.D.clamp_(min=epsilon)
        model.alpha.clamp_(min=-1e3, max=1e3)
    tqdm.write(f"epoch {i} || loss: {loss.item():.4f}, "
                f"E_2: {E_2.item():.4f} -> {n_open}, "
                f"E_3: {E_3.item():.4f} -> {n_triangles}, "                
                f"alpha mean: {torch.sigmoid(model.alpha).mean().item():.4f}")

# save the results
alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
n_round = args.nround
p_blocks = model.p_blocks.detach().cpu().numpy()
N_blocks = model.N_blocks.detach().cpu().numpy()

# mkdir if not exists
p_res = f"../fit_res/SBM_iter/{args.name}"
p_res = Path(p_res)
p_res.mkdir(exist_ok=True, parents=True)

with open(f"{p_res}/{args.dataset}.alpha_R_PB_NB", "wb") as f:
    pickle.dump((alpha, n_round, p_blocks, N_blocks), f)

# save the results in text too
with open(f"{p_res}/{args.dataset}.txt", "w") as f:
    f.write(f"{n_blocks} {n_round}\n")
    # alpha
    for alpha_ in alpha:
        f.write(f"{alpha_}\n")
    # p_blocks
    for row in p_blocks:
        for p in row:
            f.write(f"{p} ")
        f.write("\n")
    # N_blocks
    for n in N_blocks:
        f.write(f"{n}\n")
