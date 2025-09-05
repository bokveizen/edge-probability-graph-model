import os
from pathlib import Path
import torch
import argparse
import pickle
import networkx as nx
from collections import Counter
from tqdm import tqdm, trange
from kron_seeds import k_dict, seed_matrix_dict

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

def isolated_nodes(edge_probs: torch.tensor, edge_probs_rem: torch.tensor, sample_probs: torch.tensor, num_rounds: int):
    # edge probs: (n, n); sample probs: (n, )
    # compute the expected number of isolated nodes
    
    assert edge_probs.shape[0] == edge_probs.shape[1] == sample_probs.shape[0], "edge_probs and sample_probs should have the same shape"
    # n_nodes = edge_probs.shape[0]
    
    # sort each row of edge_probs in descending order
    # edge_probs_sorted: (n, n)
    # order: (n, n); each row is a permutation of [0, 1, ..., n - 1]
    # breakpoint()
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

# independent non-uniform node sampling
# optimize degrees too
# the grouping is kept
# nodes with the same degree are sampled with the same probability (i.e., the same alpha)

# make a factorial table
factorial_table = torch.tensor([1, 1, 2, 6, 24, 
                                120, 720, 5040, 40320, 362880, 
                                3628800, 39916800, 479001600, 6227020800, 87178291200,
                                1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000,
                                2432902008176640000])
factorial_table = factorial_table.to(device)

def count_zero(x: int, k: int):
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return k - count


num_eps = 1e-8
class iidBindingKroneckerJoint(torch.nn.Module):
    def __init__(self, alpha: torch.tensor, n_round: int, seed_matrix: torch.tensor, k: int):
        super().__init__()
        self.alpha = torch.nn.Parameter(alpha.to(device))  # k + 1
        self.n_round = torch.tensor(float(n_round)).to(device)  # scalar
        self.n_round_int = n_round
        # self.seed_matrix = torch.tensor(seed_matrix).to(device)  # (d, d)
        self.seed_matrix_orig = seed_matrix.to(device)  # (d, d), usually (2, 2)
        self.seed_sum_orig = self.seed_matrix_orig.sum()
        # make seed_matrix a parameter
        self.seed_00 = torch.nn.Parameter(self.seed_matrix_orig[0, 0])
        self.seed_01 = torch.nn.Parameter(self.seed_matrix_orig[0, 1])
        self.seed_11 = torch.nn.Parameter(self.seed_matrix_orig[1, 1])                
        # self.seed_00 = self.seed_matrix_orig[0, 0]
        # self.seed_01 = self.seed_matrix_orig[0, 1]
        # self.seed_11 = self.seed_matrix_orig[1, 1]               
        
        self.d = self.seed_matrix_orig.shape[0]  # the size of seed matrix
        # self.k = torch.tensor(float(k)).to(device)  # scalar
        self.k = k  # scalar
        # the number of nodes with i 0's, which is k choose i
        self.ni = torch.tensor([float(factorial_table[self.k]) / (factorial_table[i] * factorial_table[self.k - i]) for i in range(self.k + 1)]).to(device)  # (k + 1)
        self.prepare_once()
        
    def prepare_once(self):
        # create a 1d index tensor "zeor_index" for the nodes 0 <= v <= 2^k - 1, where zero_index[v] = # of zeros in v
        # zero_index: (2^k, )
        self.zero_index = torch.zeros(2 ** self.k, dtype=torch.long).to(device)
        for v in range(2 ** self.k):
            # self.zero_index[v] = bin(v).count("0") - 1
            self.zero_index[v] = count_zero(v, self.k)
        
        self.indices_seq = []
        for x0 in trange(self.k + 1, desc="x0"):
            x1 = self.k - x0
            for x00 in range(x0 + 1):
                x01 = x0 - x00
                for x10 in range(x1 + 1):
                    x11 = x1 - x10
                    for x000 in range(x00 + 1):
                        x001 = x00 - x000
                        for x010 in range(x01 + 1):
                            x011 = x01 - x010
                            for x100 in range(x10 + 1):
                                x101 = x10 - x100
                                for x110 in range(x11 + 1):
                                    x111 = x11 - x110
                                    if x01 == 0 and x10 == 0:
                                        continue
                                    if x001 == 0 and x011 == 0 and x100 == 0 and x110 == 0:
                                        continue
                                    if x001 == 0 and x010 == 0 and x101 == 0 and x110 == 0:
                                        continue
                                    self.indices_seq.append([x0, x1, x00, x01, x10, x11, x000, x001, x010, x011, x100, x101, x110, x111])
        self.n_indices = len(self.indices_seq)
        self.indices_seq = torch.tensor(self.indices_seq).to(device)
        self.x0_seq = self.indices_seq[:, 0]
        self.x1_seq = self.indices_seq[:, 1]
        self.x00_seq = self.indices_seq[:, 2]
        self.x01_seq = self.indices_seq[:, 3]
        self.x10_seq = self.indices_seq[:, 4]
        self.x11_seq = self.indices_seq[:, 5]
        self.x000_seq = self.indices_seq[:, 6]
        self.x001_seq = self.indices_seq[:, 7]
        self.x010_seq = self.indices_seq[:, 8]
        self.x011_seq = self.indices_seq[:, 9]
        self.x100_seq = self.indices_seq[:, 10]
        self.x101_seq = self.indices_seq[:, 11]
        self.x110_seq = self.indices_seq[:, 12]
        self.x111_seq = self.indices_seq[:, 13]
                
        # self.r12_indices = self.indices_seq[:, [2, 3, 4, 5]]
        self.r12_indices = torch.stack((self.x00_seq, self.x01_seq, self.x10_seq, self.x11_seq), dim=1)
        x000_010 = self.x000_seq + self.x010_seq
        x001_011 = self.x001_seq + self.x011_seq
        x100_110 = self.x100_seq + self.x110_seq
        x101_111 = self.x101_seq + self.x111_seq
        self.r13_indices = torch.stack((x000_010, x001_011, x100_110, x101_111), dim=1)
        x000_100 = self.x000_seq + self.x100_seq
        x001_101 = self.x001_seq + self.x101_seq
        x010_110 = self.x010_seq + self.x110_seq
        x011_111 = self.x011_seq + self.x111_seq
        self.r23_indices = torch.stack((x000_100, x001_101, x010_110, x011_111), dim=1)
        
        # zero1 = x0
        # zero2 = x00 + x10
        # zero3 = x000 + x010 + x100 + x110   
        self.zero1_seq = self.x0_seq
        self.zero2_seq = self.x00_seq + self.x10_seq
        self.zero3_seq = self.x000_seq + self.x010_seq + self.x100_seq + self.x110_seq
        
        self.indices = torch.arange(self.k + 1).to(device)
        self.indices_meshgrid = torch.meshgrid(self.indices, self.indices, self.indices, self.indices)
        
        self.n_triplet_seq = factorial_table[self.k] / (factorial_table[self.x000_seq] * factorial_table[self.x001_seq] * factorial_table[self.x010_seq] *\
            factorial_table[self.x011_seq] * factorial_table[self.x100_seq] * factorial_table[self.x101_seq] * factorial_table[self.x110_seq] * factorial_table[self.x111_seq])
        self.n_triplet_seq /= 6    
    
    def prepare(self):
        # normalization to maintain the number of edges
        # with torch.no_grad():
        seed_matrix_sum = self.seed_00 + 2 * self.seed_01 + self.seed_11
        seed_matrix_ratio = self.seed_sum_orig / seed_matrix_sum
        # self.seed_00 *= self.seed_sum_orig / seed_matrix_sum
        # self.seed_01 *= self.seed_sum_orig / seed_matrix_sum
        # self.seed_11 *= self.seed_sum_orig / seed_matrix_sum
        self.seed_matrix = torch.zeros_like(self.seed_matrix_orig).to(device)
        self.seed_matrix[0, 0] += self.seed_00 * seed_matrix_ratio
        self.seed_matrix[0, 1] += self.seed_01 * seed_matrix_ratio
        self.seed_matrix[1, 0] += self.seed_01 * seed_matrix_ratio
        self.seed_matrix[1, 1] += self.seed_11 * seed_matrix_ratio
        self.seed_matrix.clamp_(min=1e-4, max=1 - 1e-4)
        self.exp_edges = self.seed_matrix.sum() ** self.k  # scalar
        
        print(self.seed_matrix)
        
        self.n_total = 2 ** self.k

        # compute the k-th Kronecker power of seed_matrix
        self.P_final = torch.zeros_like(self.seed_matrix).to(device)
        self.P_final += self.seed_matrix        
        for i1 in range(self.k - 1):
            self.P_final = torch.kron(self.P_final, self.seed_matrix)

        
        i1, i2, i3, i4 = self.indices_meshgrid
        self.P = (self.seed_matrix[0, 0] ** i1) * (self.seed_matrix[0, 1] ** i2) * (self.seed_matrix[1, 0] ** i3) * (self.seed_matrix[1, 1] ** i4)                        
                
        # for each triplet, record the three probabilities
        # (n_indices)
        self.p12 = self.P[tuple(self.r12_indices.T)]
        self.p13 = self.P[tuple(self.r13_indices.T)]
        self.p23 = self.P[tuple(self.r23_indices.T)]
        
        # (n_indices, 3)
        self.p_stack = torch.stack((self.p12, self.p13, self.p23), dim=1)
        self.R_num = (1 - (1 - self.p_stack) ** (1 / self.n_round))
        self.R_num_full = (1 - (1 - self.P_final) ** (1 / self.n_round))

    def forward(self):
        self.prepare()
        # kr_power = self.k
        # s = self.seed_matrix
        # indices = self.indices
        # i1, i2, i3, i4 = self.indices_meshgrid
        alpha_sigmoid = torch.sigmoid(self.alpha)  # k + 1; alpha_sigmoid[i] = the alpha for nodes with i zeros
        alpha1 = alpha_sigmoid[self.zero1_seq]
        alpha2 = alpha_sigmoid[self.zero2_seq]
        alpha3 = alpha_sigmoid[self.zero3_seq]
        alpha_stack = torch.stack((alpha3, alpha2, alpha1), dim=1)
        
        # alpha_full: (2^k, )
        alpha_full = alpha_sigmoid[self.zero_index]
        # Q_full: (2^k, 2^k)
        Q_full = torch.outer(alpha_full, alpha_full)
        # R_full: (2^k, 2^k)
        R_full = self.R_num_full / (Q_full + num_eps)
        R_full.clamp_(max=1)
        # Prem_full: (2^k, 2^k)
        Prem_full = 1 - ((1 - self.P_final) / (1 - Q_full) ** self.n_round + num_eps)
        Prem_full.clamp_(min=0)
        # zero diagonal
        Prem_full[torch.arange(self.n_total), torch.arange(self.n_total)] = 0
        
        P = self.p_stack
        
        q12 = alpha1 * alpha2
        q13 = alpha1 * alpha3
        q23 = alpha2 * alpha3
        # Q, R, P_Rem: (n_indices, 3)
        Q = torch.stack((q12, q13, q23), dim=1)        
        r_stack = self.R_num / Q
        r_stack.clamp_(max=1)
        prem_stack = 1 - ((1 - P) / (1 - Q) ** self.n_round + num_eps)
        prem_stack.clamp_(min=0)
        
        # E0_local = torch.zeros(kr_power + 1).to(device)
        # E1_local = torch.zeros(kr_power + 1).to(device)
        # E2_local = torch.zeros(kr_power + 1).to(device)
        # E3_local = torch.zeros(kr_power + 1).to(device)
        
        E0 = torch.tensor(0.).to(device)
        E1 = torch.tensor(0.).to(device)
        E2 = torch.tensor(0.).to(device)
        E3 = torch.tensor(0.).to(device)
        
        n_indices = self.n_indices
        # sorted each row of r_stack and record the order
        # r_stack_sorted: (n_indices, 3)
        # order: (n_indices, 3); each row is a permutation of [0, 1, 2]
        r_stack_sorted, order = torch.sort(r_stack, dim=1)
        prem_stack_sorted = prem_stack[torch.arange(n_indices).unsqueeze(1), order]
        alpha_stack_sorted = alpha_stack[torch.arange(n_indices).unsqueeze(1), order]
        
        rmin = r_stack_sorted[:, 0]
        rmid = r_stack_sorted[:, 1]
        rmax = r_stack_sorted[:, 2]
        
        prem_min = prem_stack_sorted[:, 0]
        prem_mid = prem_stack_sorted[:, 1]
        prem_max = prem_stack_sorted[:, 2]
        
        # e.g., alpha_min is the alpha for the node NOT in rmin
        alpha_min = alpha_stack_sorted[:, 0]
        alpha_mid = alpha_stack_sorted[:, 1]
        alpha_max = alpha_stack_sorted[:, 2]
        
        pg_3 = alpha_min * alpha_mid * alpha_max
        # the two endpoints of rmin sampled
        pg_2_min = (1 - alpha_min) * alpha_mid * alpha_max
        pg_2_mid = alpha_min * (1 - alpha_mid) * alpha_max
        pg_2_max = alpha_min * alpha_mid * (1 - alpha_max)
        
        p_round_3 = pg_3 * rmin        
        p_round_2 = pg_3 * (rmid - rmin)  # rmax and rmid are sampled
        p_round_1_min = pg_2_min * rmin
        p_round_1_mid = pg_2_mid * rmid
        p_round_1_max = pg_2_max * rmax + pg_3 * (rmax - rmid)  # rmax is sampled
        p_round_0 = 1 - p_round_3 - p_round_2 - p_round_1_min - p_round_1_mid - p_round_1_max
        
        p_round_le1_min = p_round_0 + p_round_1_min
        p_round_le1_mid = p_round_0 + p_round_1_mid
        p_round_le1_max = p_round_0 + p_round_1_max
        p_round_le2_min_mid = p_round_0 + p_round_1_min + p_round_1_mid
        p_round_le2_min_max = p_round_0 + p_round_1_min + p_round_1_max
        p_round_le2_mid_max = p_round_0 + p_round_1_mid + p_round_1_max + p_round_2
        
        # accumulate the expected number of triplets
        P_0 = (p_round_0 ** self.n_round) * (1 - prem_min) * (1 - prem_mid) * (1 - prem_max)
        P_1_min = (
            (p_round_0 ** self.n_round) * prem_min * (1 - prem_mid) * (1 - prem_max) +
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * (1 - prem_mid) * (1 - prem_max)
        )
        P_1_mid = (
            (p_round_0 ** self.n_round) * prem_mid * (1 - prem_min) * (1 - prem_max) +
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * (1 - prem_min) * (1 - prem_max)
        )
        P_1_max = (
            (p_round_0 ** self.n_round) * prem_max * (1 - prem_min) * (1 - prem_mid) +
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * (1 - prem_min) * (1 - prem_mid)
        )
        P_2_min_mid = (
            (p_round_0 ** self.n_round) * prem_min * prem_mid * (1 - prem_max) +
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * prem_mid * (1 - prem_max) +
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * prem_min * (1 - prem_max) +
            (p_round_le2_min_mid ** self.n_round - p_round_le1_min ** self.n_round - p_round_le1_mid ** self.n_round + p_round_0 ** self.n_round) * (1 - prem_max)
        )
        P_2_min_max = (
            (p_round_0 ** self.n_round) * prem_min * prem_max * (1 - prem_mid) +
            (p_round_le1_min ** self.n_round - p_round_0 ** self.n_round) * prem_max * (1 - prem_mid) +
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * prem_min * (1 - prem_mid) +
            (p_round_le2_min_max ** self.n_round - p_round_le1_min ** self.n_round - p_round_le1_max ** self.n_round + p_round_0 ** self.n_round) * (1 - prem_mid)
        )
        P_2_mid_max = (
            (p_round_0 ** self.n_round) * prem_mid * prem_max * (1 - prem_min) +
            (p_round_le1_mid ** self.n_round - p_round_0 ** self.n_round) * prem_max * (1 - prem_min) +
            (p_round_le1_max ** self.n_round - p_round_0 ** self.n_round) * prem_mid * (1 - prem_min) +
            (p_round_le2_mid_max ** self.n_round - p_round_le1_mid ** self.n_round - p_round_le1_max ** self.n_round + p_round_0 ** self.n_round) * (1 - prem_min)
        )                
        P_3 = 1 - P_0 - P_1_min - P_1_mid - P_1_max - P_2_min_mid - P_2_min_max - P_2_mid_max
        P_1 = P_1_min + P_1_mid + P_1_max
        P_2 = P_2_min_mid + P_2_min_max + P_2_mid_max
        
        E0 += (self.n_triplet_seq * P_0).sum()
        E1 += (self.n_triplet_seq * P_1).sum()
        E2 += (self.n_triplet_seq * P_2).sum()
        E3 += (self.n_triplet_seq * P_3).sum()
                
        n_isolated = isolated_nodes(R_full, Prem_full, alpha_full, self.n_round_int)
        n_nonisolated = self.n_total - n_isolated        
        
        # breakpoint()        
        return E0, E1, E2, E3, n_nonisolated

# inverse sigmoid
def inv_sigmoid(x: float):
    return torch.log(x / (1 - x))

seed_matrix = torch.tensor(seed_matrix_dict[args.dataset]).to(device)
# bound to [0, 1]
seed_matrix.clamp_(min=0, max=1)

# seed_matrix = torch.tensor(
#     [
#         [0.658491, 0.714804],
#         [0.714351, 0.0595782],
#     ]
# ).to(device)

# make it symmetric
seed_matrix = (seed_matrix + seed_matrix.T) / 2

p_data = Path("../data/nx_graph")

with open(p_data / f"{args.dataset}.graph", "rb") as f:
    graph: nx.Graph = pickle.load(f)

n_gt = graph.number_of_nodes()
m_gt = graph.number_of_edges()
deg_list = [len(graph[v]) for v in graph]
Em_orig = graph.number_of_edges()
n_triangles = sum(nx.triangles(graph).values())
n_wedges = sum(d * (d - 1) / 2 for d in deg_list)
n_open = n_wedges - n_triangles
n_triangles /= 3
k = k_dict[args.dataset]

alpha_raw = [args.alpha for _ in range(k + 1)]
alpha_inv_sigmoid = inv_sigmoid(torch.tensor(alpha_raw))

model = iidBindingKroneckerJoint(alpha_inv_sigmoid, args.nround, seed_matrix, k)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epsilon = 1e-3
loss_eps = 1e-16
ratio_gt = n_triangles / n_open
for i in tqdm(range(args.ep), desc="Training", unit="epoch"):
    optimizer.zero_grad()
    E_0, E_1, E_2, E_3, E_n = model()
    ratio_gen = E_3 / E_2
    loss_triangle = (1 - (E_3 / n_triangles)) ** 2
    loss_wedge = (1 - (E_2 / n_open)) ** 2
    loss_ratio = (1 - (ratio_gen / ratio_gt)) ** 2
    loss_node = (1 - (E_n / n_gt)) ** 2
    loss = args.wt * loss_triangle + args.ww * loss_wedge + args.wr * loss_ratio + args.wn * loss_node
    # loss = loss_triangle + loss_wedge + loss_nodes
    loss = loss_triangle + loss_wedge
    if loss < loss_eps:
        break
    loss.backward()
    for p in model.parameters():
        if torch.isnan(p.grad).any():
            p.grad = torch.where(torch.isnan(p.grad), torch.zeros_like(p.grad), p.grad)
    # grad climping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)        
    optimizer.step()
    with torch.no_grad():        
        model.alpha.clamp_(min=-1e3, max=1e3)
        # make sure seed00, seed01, and seed11 are between 0 and 1
        model.seed_00.clamp_(min=1e-4, max=1 - 1e-4)
        model.seed_01.clamp_(min=1e-4, max=1 - 1e-4)
        model.seed_11.clamp_(min=1e-4, max=1 - 1e-4)
        # normalize the seed matrix
        seed_matrix_sum = model.seed_00 + 2 * model.seed_01 + model.seed_11
        model.seed_00 *= model.seed_sum_orig / seed_matrix_sum
        model.seed_01 *= model.seed_sum_orig / seed_matrix_sum
        model.seed_11 *= model.seed_sum_orig / seed_matrix_sum
    tqdm.write(f"epoch {i} || loss: {loss.item():.4f}, "
                f"E_2: {E_2.item():.4f} -> {n_open}, "
                f"E_3: {E_3.item():.4f} -> {n_triangles}, "
                # f"E_m: {E_m.item():.4f} -> {m_gt}, "
                f"E_n: {E_n.item():.4f} -> {n_gt}, "
                f"alpha mean: {torch.sigmoid(model.alpha).mean().item():.4f}")

# save the results
alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
with torch.no_grad():
    seed_matrix_sum = model.seed_00 + 2 * model.seed_01 + model.seed_11
    model.seed_00 *= model.seed_sum_orig / seed_matrix_sum
    model.seed_01 *= model.seed_sum_orig / seed_matrix_sum
    model.seed_11 *= model.seed_sum_orig / seed_matrix_sum                        
model.seed_matrix = torch.zeros_like(model.seed_matrix_orig).to(device)
model.seed_matrix[0, 0] += model.seed_00
model.seed_matrix[0, 1] += model.seed_01
model.seed_matrix[1, 0] += model.seed_01
model.seed_matrix[1, 1] += model.seed_11 
k = model.k
n_round = model.n_round_int
seed_matrix = model.seed_matrix.detach().cpu().numpy()

# mkdir if not exists
p_res = f"../fit_res/KR_iid_joint/{args.name}"
p_res = Path(p_res)
p_res.mkdir(exist_ok=True, parents=True)

with open(f"{p_res}/{args.dataset}.alpha_seed_nround", "wb") as f:
    pickle.dump((alpha, seed_matrix, k, n_round), f)
    
# and save in a text file
with open(f"{p_res}/{args.dataset}.txt", "w") as f:
    f.write(f"{k} {n_round} {seed_matrix.shape[0]}\n")
    # write the alpha sequence
    for a in alpha:
        f.write(f"{a:.16f}\n")
    # write the seed matrix
    for row in seed_matrix:
        for e in row:
            f.write(f"{e:.16f} ")
        f.write("\n")
