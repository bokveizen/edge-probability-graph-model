import os
from pathlib import Path
import numpy as np
import torch
import argparse
import pickle
import networkx as nx
from collections import Counter
from tqdm import tqdm

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
parser.add_argument("--wn", type=float, default=0.0, help="weight of nodes")
parser.add_argument("--name", type=str, default="test", help="output folder suffix")
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


def isolated_nodes(edge_probs: torch.tensor, edge_probs_rem: torch.tensor, sample_probs: torch.tensor, num_rounds: int):
    # edge probs: (n, n); sample probs: (n, )
    # compute the expected number of isolated nodes
    
    assert edge_probs.shape[0] == edge_probs.shape[1] == sample_probs.shape[0], "edge_probs and sample_probs should have the same shape"
    # n_nodes = edge_probs.shape[0]
    
    # sort each row of edge_probs in descending order
    # edge_probs_sorted: (n, n)
    # order: (n, n); each row is a permutation of [0, 1, ..., n - 1]
    edge_probs_sorted, order = torch.sort(edge_probs, dim=1, descending=True)
    # sample_probs_sorted: (n, n)
    # each row is a permutation of sampled_probs following the order of edge_probs_sorted
    # sample_probs_sorted = sample_probs[torch.arange(n_nodes).unsqueeze(1), order]
    sample_probs_sorted = sample_probs[order]
    
    # coef_sorted: (n, n)
    # coef_sorted[i, j] = (\prod_{k = 0}^{j-1} (1 - sample_probs_sorted[i, k])) sample_probs_sorted[i, j]
    sampled_probs_cumprod = torch.cumprod(1 - sample_probs_sorted, dim=1)
    # sampled_probs_cumprod[i, j] = \prod_{k = 0}^{j} (1 - sample_probs_sorted[i, k])
    # left shift sampled_probs_cumprod by 1
    sampled_probs_cumprod = torch.cat((torch.ones(sampled_probs_cumprod.shape[0], 1).to(device), sampled_probs_cumprod[:, :-1]), dim=1)
    # sampled_probs_cumprod[i, j] = \prod_{k = 0}^{j-1} (1 - sample_probs_sorted[i, k])
    
    coef_sorted = sampled_probs_cumprod * sample_probs_sorted
    # coef_sorted[i, j] = \prod_{k = 0}^{j-1} (1 - sample_probs_sorted[i, k]) sample_probs_sorted[i, j]
    
    # pc: (n, )
    # pc[i] = sample_probs[i] * (\sum_j coef_sorted[i, j] * edge_probs_sorted[i, j])
    pc = (coef_sorted * edge_probs_sorted).sum(dim=1)
    pc = pc * sample_probs
    
    # pi: (n, )
    # pi[i] = (1 - pc[i])^num_rounds
    pi = (1 - pc) ** num_rounds
    
    # now consider probs_rem: (n, n)
    p_rem_inv = 1 - edge_probs_rem
    pi_rem = torch.prod(p_rem_inv, dim=1)
    pi = pi * pi_rem
    
    return pi.sum()


num_eps = 1e-8
class iidBindingSbm(torch.nn.Module):
    def __init__(self, alpha: torch.tensor, n_round: int, p_blocks: torch.tensor, N_blocks: torch.tensor):
        # alpha: (n_blocks, ); the binding strength of the nodes in each block
        # n_round: number of rounds
        # p_blocks: (n_blocks, n_blocks); p_blocks[i, j] is the probability of an edge between a node in block i and a node in block j
        # N_blocks: (n_blocks, ); N_blocks[i] is the number of nodes in block i
        
        super().__init__()
        # make alpha a tensor of parameters
        self.alpha = torch.nn.Parameter(alpha.to(device))
        self.n_round = torch.tensor(float(n_round)).to(device)
        self.n_round_int = n_round
        self.p_blocks = p_blocks.to(device)
        self.N_blocks = N_blocks.to(device)
        self.n_total = N_blocks.sum().int().item()
        
        assert self.alpha.shape[0] == self.p_blocks.shape[0] == self.p_blocks.shape[1] == self.N_blocks.shape[0], "the number of blocks is not consistent"
        self.n_blocks = self.alpha.shape[0]
        
        # B_index is a list of indices, has N_block[i] elements with value i, and so on
        self.B_index = []
        for i in range(self.n_blocks):
            self.B_index += [i for _ in range(self.N_blocks[i])]
        self.B_index = torch.tensor(self.B_index).to(device)
        self.mesh = torch.meshgrid(torch.arange(n_blocks), torch.arange(n_blocks), torch.arange(n_blocks))
                
    
    def forward(self):                        
        P = self.p_blocks        
        NB = self.N_blocks
        n_blocks = self.n_blocks
        alpha_sigmoid = torch.sigmoid(self.alpha)
        
        alpha_full = alpha_sigmoid[self.B_index]
        # expand P to (n_total, n_total) using B_index
        P_full = P[self.B_index, :][:, self.B_index]        
        # zero diagonal
        P_full[torch.arange(self.n_total), torch.arange(self.n_total)] = 0
                
        # Q: (n_blocks, n_blcosk); Q[i, j] = alpha_sigmoid[i] * alpha_sigmoid[j]
        Q = torch.outer(alpha_sigmoid, alpha_sigmoid)
        Q_full = torch.outer(alpha_full, alpha_full)
        
        i, j, k = self.mesh
        
        R = (1 - (1 - P) ** (1 / self.n_round)) / (Q + num_eps)
        R.clamp_(min=0, max=1)
        R_full = (1 - (1 - P_full) ** (1 / self.n_round)) / (Q_full + num_eps)
        R_full.clamp_(min=0, max=1)
        R_full[torch.arange(self.n_total), torch.arange(self.n_total)] = 0
        
        P_rem = 1 - ((1 - P) / ((1 - Q) ** self.n_round + num_eps))        
        P_rem.clamp_(min=0, max=1)
        P_rem_full = 1 - ((1 - P_full) / ((1 - Q_full) ** self.n_round + num_eps))
        P_rem_full.clamp_(min=0, max=1)
        P_rem_full[torch.arange(self.n_total), torch.arange(self.n_total)] = 0
        n_isolated = isolated_nodes(R_full, P_rem_full, alpha_full, self.n_round_int)
        n_nonisolated = self.n_total - n_isolated
        E_m = P_full.sum() * 0.5
        
        # R_stack: (n_blocks, n_blocks, n_blocks, 3); R_stack[i, j, k] = [R[i, j], R[i, k], R[j, k]]
        R_stack = torch.stack((R[i, j], R[i, k], R[j, k]), dim=-1)
        
        # R_max: (n_blocks, n_blocks, n_blocks); R_max[i, j, k] = max(R[i, j], R[i, k], R[j, k])
        # R_max_indices: (n_blocks, n_blocks, n_blocks); R_max_indices[i, j, k] = argmax(R[i, j], R[i, k], R[j, k])
        # similar for "mid" and "min"
        R_sorted, indices = torch.sort(R_stack, dim=-1)
        R_min, R_min_indices = R_sorted[..., 0], indices[..., 0]
        R_mid, R_mid_indices = R_sorted[..., 1], indices[..., 1]
        R_max, R_max_indices = R_sorted[..., 2], indices[..., 2]
        
        # similar for P_rem
        P_rem_stack = torch.stack((P_rem[i, j], P_rem[i, k], P_rem[j, k]), dim=-1)
        # construct P_rem_max, P_rem_mid, and P_rem_min according to R_max_indices, R_mid_indices, and R_min_indices
        # P_rem_max: (n_blocks, n_blocks, n_blocks); P_rem_max[i, j, k] = [P_rem[i, j], P_rem[i, k], P_rem[j, k]][R_max_indices[i, j, k]]
        # similar for "mid" and "min"
        P_rem_max = torch.gather(P_rem_stack, dim=-1, index=R_max_indices.unsqueeze(-1)).squeeze(-1)
        P_rem_mid = torch.gather(P_rem_stack, dim=-1, index=R_mid_indices.unsqueeze(-1)).squeeze(-1)
        P_rem_min = torch.gather(P_rem_stack, dim=-1, index=R_min_indices.unsqueeze(-1)).squeeze(-1)               
        
        # similar for alpha_sigmoid
        # alpha_max: (n_blocks, n_blocks, n_blocks)
        # alpha_max[i, j, k] = [alpha_sigmoid[k], alpha_sigmoid[j], alpha_sigmoid[i]][R_max_indices[i, j, k]]
        # alpha_max[i, j, k] is alpha[k] if R_max is R[i, j]
        # alpha_max[i, j, k] is alpha[j] if R_max is R[i, k]
        # alpha_max[i, j, k] is alpha[i] if R_max is R[j, k]
        alpha_sigmoid_stack = torch.stack((alpha_sigmoid[k], alpha_sigmoid[j], alpha_sigmoid[i]), dim=-1)        
        alpha_max = torch.gather(alpha_sigmoid_stack, dim=-1, index=R_max_indices.unsqueeze(-1)).squeeze(-1)
        alpha_mid = torch.gather(alpha_sigmoid_stack, dim=-1, index=R_mid_indices.unsqueeze(-1)).squeeze(-1)
        alpha_min = torch.gather(alpha_sigmoid_stack, dim=-1, index=R_min_indices.unsqueeze(-1)).squeeze(-1)
                
        # pg_3: (n_blocks, n_blocks, n_blocks); pg_3[i, j, k] = alpha_sigmoid[i] * alpha_sigmoid[j] * alpha_sigmoid[k]        
        pg_3 = alpha_max * alpha_mid * alpha_min
        # pg_2_xxx: (n_blocks, n_blocks, n_blocks)
        # pg_2_max_mid[i, j, k] is the probability that the two endpoints of R_min[i, j, k] are sampled       
        # pg_2_max_min[i, j, k] is the probability that the two endpoints of R_mid[i, j, k] are sampled
        # pg_2_mid_min[i, j, k] is the probability that the two endpoints of R_max[i, j, k] are sampled
        pg_2_max = (1 - alpha_max) * alpha_mid * alpha_min  # corresponding to R_max
        pg_2_mid = alpha_max * (1 - alpha_mid) * alpha_min  # corresponding to R_mid
        pg_2_min = alpha_max * alpha_mid * (1 - alpha_min)  # corresponding to R_min

        # p_round: (n_blocks, n_blocks, n_blocks)
        p_round_3 = pg_3 * R_min
        p_round_2 = pg_3 * (R_mid - R_min)  # max and mid
        p_round_1_max = pg_3 * (R_max - R_mid) + pg_2_max * R_max
        p_round_1_mid = pg_2_mid * R_mid
        p_round_1_min = pg_2_min * R_min
        p_round_0 = 1 - p_round_3 - p_round_2 - p_round_1_max - p_round_1_mid - p_round_1_min
        p_round_le1_max = p_round_0 + p_round_1_max
        p_round_le1_mid = p_round_0 + p_round_1_mid
        p_round_le1_min = p_round_0 + p_round_1_min
        p_round_le2_max_mid = p_round_0 + p_round_1_max + p_round_1_mid + p_round_2
        p_round_le2_max_min = p_round_0 + p_round_1_max + p_round_1_min
        p_round_le2_mid_min = p_round_0 + p_round_1_mid + p_round_1_min
                
        # p_motifs: (n_blocks, n_blocks, n_blocks)
        P_0 = (p_round_0 ** self.n_round) * (1 - P_rem_max) * (1 - P_rem_mid) * (1 - P_rem_min)
        P_1_max = (
            (p_round_0 ** self.n_round) * P_rem_max * (1 - P_rem_mid) * (1 - P_rem_min) + 
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * (1 - P_rem_mid) * (1 - P_rem_min)
            )
        P_1_mid = (
            (p_round_0 ** self.n_round) * P_rem_mid * (1 - P_rem_min) * (1 - P_rem_max) + 
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * (1 - P_rem_min) * (1 - P_rem_max)
            )
        P_1_min = (
            (p_round_0 ** self.n_round) * P_rem_min * (1 - P_rem_max) * (1 - P_rem_mid) + 
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * (1 - P_rem_max) * (1 - P_rem_mid)
            )
        P_2_max_mid = (
            (p_round_0 ** self.n_round) * P_rem_max * P_rem_mid * (1 - P_rem_min) + 
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * P_rem_mid * (1 - P_rem_min) + 
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * P_rem_max * (1 - P_rem_min) + 
            (p_round_le2_max_mid ** self.n_round - p_round_le1_max ** self.n_round - p_round_le1_mid ** self.n_round + p_round_0 ** self.n_round) * (1 - P_rem_min)
            )
        P_2_max_min = (
            (p_round_0 ** self.n_round) * P_rem_max * P_rem_min * (1 - P_rem_mid) + 
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * P_rem_min * (1 - P_rem_mid) + 
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * P_rem_max * (1 - P_rem_mid) + 
            (p_round_le2_max_min ** self.n_round - p_round_le1_max ** self.n_round - p_round_le1_min ** self.n_round + p_round_0 ** self.n_round) * (1 - P_rem_mid)
            )
        P_2_mid_min = (
            (p_round_0 ** self.n_round) * P_rem_mid * P_rem_min * (1 - P_rem_max) + 
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * P_rem_min * (1 - P_rem_max) + 
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * P_rem_mid * (1 - P_rem_max) + 
            (p_round_le2_mid_min ** self.n_round - p_round_le1_mid ** self.n_round - p_round_le1_min ** self.n_round + p_round_0 ** self.n_round) * (1 - P_rem_max)
            )
        P_3 = 1 - P_0 - P_1_max - P_1_mid - P_1_min - P_2_max_mid - P_2_max_min - P_2_mid_min
        P_1 = P_1_max + P_1_mid + P_1_min
        P_2 = P_2_max_mid + P_2_max_min + P_2_mid_min
        
        # compute the number of triplets
        # n_triplets: (n_blocks, n_blocks, n_blocks)
        # if i, j, and k are all distinct, n_triplets[i, j, k] = NB[i] * NB[j] * NB[k] / 6 
        # if i == j != k, n_triplets[i, j, k] = NB[i] * (NB[i] - 1) * NB[k] / 6
        # if i == k != j, n_triplets[i, j, k] = NB[i] * (NB[i] - 1) * NB[j] / 6
        # if j == k != i, n_triplets[i, j, k] = NB[j] * (NB[j] - 1) * NB[i] / 6
        # if i == j == k, n_triplets[i, j, k] = NB[i] * (NB[i] - 1) * (NB[i] - 2) / 6
        
        # i, j, k = torch.meshgrid(torch.arange(n_blocks), torch.arange(n_blocks), torch.arange(n_blocks))

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
        E_0 = (P_0 * n_triplets).sum()
        E_1 = (P_1 * n_triplets).sum()
        E_2 = (P_2 * n_triplets).sum()
        E_3 = (P_3 * n_triplets).sum()
        
        return E_0, E_1, E_2, E_3, E_m, n_nonisolated

p_data = Path("../data/nx_graph")
p_SBM_PBSB = Path("../data/sbm_PB_NB")

with open(p_data / f"{args.dataset}.graph", "rb") as f:
    graph: nx.Graph = pickle.load(f)

# inverse sigmoid
def inv_sigmoid(x: float):
    return torch.log(x / (1 - x))


n_gt = graph.number_of_nodes()
m_gt = graph.number_of_edges()
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

model = iidBindingSbm(alpha=alpha_inv_sigmoid, n_round=args.nround, p_blocks=p_blocks, N_blocks=N_blocks).to(device)

# optimize the model (i.e., alpha) using Adam
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epsilon = 1e-3
loss_eps = 1e-16
# ratio_gt = (n_triangles / n_open) / 3
for i in tqdm(range(args.ep), desc="Training", unit="epoch"):
    optimizer.zero_grad()
    E_0, E_1, E_2, E_3, E_m, E_n = model()
    ratio_gen = E_3 / E_2
    loss_triangle = (1 - (E_3 / n_triangles)) ** 2
    loss_wedge = (1 - (E_2 / n_open)) ** 2
    loss_ratio = (1 - (ratio_gen / ratio_gt)) ** 2
    loss_node = (1 - (E_n / n_gt)) ** 2
    loss = args.wt * loss_triangle + args.ww * loss_wedge + args.wr * loss_ratio + args.wn * loss_node
    # break if loss is small enough
    if loss < loss_eps:
        break
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
    # grad climping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)    
    optimizer.step()
    with torch.no_grad():
        # model.D.clamp_(min=epsilon)
        model.alpha.clamp_(min=-1e3, max=1e3)
    tqdm.write(f"epoch {i} || loss: {loss.item():.4f}, "
            f"E_2: {E_2.item():.4f} -> {n_open}, "
            f"E_3: {E_3.item():.4f} -> {n_triangles}, "
            f"E_m: {E_m.item():.4f} -> {m_gt}, "
            f"E_n: {E_n.item():.4f} -> {n_gt}, "
            f"alpha mean: {torch.sigmoid(model.alpha).mean().item():.4f}")

# save the results
alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
n_round = args.nround
p_blocks = model.p_blocks.detach().cpu().numpy()
N_blocks = model.N_blocks.detach().cpu().numpy()

# mkdir if not exists
p_res = f"../fit_res/SBM_iid/{args.name}"
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
