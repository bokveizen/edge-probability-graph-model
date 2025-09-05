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
class iterBindingKronecker(torch.nn.Module):
    def __init__(self, alpha: torch.tensor, n_round: int, seed_matrix: torch.tensor, k: int):
        super().__init__()
        self.alpha = torch.nn.Parameter(alpha.to(device))  # k + 1
        self.n_round = torch.tensor(float(n_round)).to(device)  # scalar
        self.n_round_int = n_round
        # self.seed_matrix = torch.tensor(seed_matrix).to(device)  # (d, d)
        self.seed_matrix = seed_matrix.to(device)  # (d, d), usually (2, 2)
        self.d = self.seed_matrix.shape[0]  # the size of seed matrix
        # self.k = torch.tensor(float(k)).to(device)  # scalar
        self.k = k  # scalar
        # the number of nodes with i 0's, which is k choose i
        self.ni = torch.tensor([float(factorial_table[self.k]) / (factorial_table[i] * factorial_table[self.k - i]) for i in range(self.k + 1)]).to(device)  # (k + 1)
        self.exp_edges = self.seed_matrix.sum() ** self.k  # scalar
        
        self.n_total = 2 ** self.k

        # compute the k-th Kronecker power of seed_matrix
        self.P_final = self.seed_matrix.clone()
        for i1 in range(self.k - 1):
            self.P_final = torch.kron(self.P_final, self.seed_matrix)

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
        i1, i2, i3, i4 = self.indices_meshgrid
        self.P = (self.seed_matrix[0, 0] ** i1) * (self.seed_matrix[0, 1] ** i2) * (self.seed_matrix[1, 0] ** i3) * (self.seed_matrix[1, 1] ** i4)        
        
        # length = self.indices_seq.shape[0], each entry is
        # n_triplet = factorial_table[kr_power] / (factorial_table[x000] * factorial_table[x001] * factorial_table[x010] *\
        # factorial_table[x011] * factorial_table[x100] * factorial_table[x101] * factorial_table[x110] * factorial_table[x111])
        self.n_triplet_seq = factorial_table[self.k] / (factorial_table[self.x000_seq] * factorial_table[self.x001_seq] * factorial_table[self.x010_seq] *\
            factorial_table[self.x011_seq] * factorial_table[self.x100_seq] * factorial_table[self.x101_seq] * factorial_table[self.x110_seq] * factorial_table[self.x111_seq])
        self.n_triplet_seq /= 6
        
        # for each triplet, record the three probabilities
        # (n_indices)
        self.p12 = self.P[tuple(self.r12_indices.T)]
        self.p13 = self.P[tuple(self.r13_indices.T)]
        self.p23 = self.P[tuple(self.r23_indices.T)]
        
        # (n_indices, 3)
        self.p_stack = torch.stack((self.p12, self.p13, self.p23), dim=1)
        # self.R_num = (1 - (1 - self.p_stack) ** (1 / self.n_round))
        # self.R_num_full = (1 - (1 - self.P_final) ** (1 / self.n_round))

    def forward(self):
        # kr_power = self.k
        # s = self.seed_matrix
        # indices = self.indices
        # i1, i2, i3, i4 = self.indices_meshgrid
        alpha_sigmoid = torch.sigmoid(self.alpha)  # k + 1; alpha_sigmoid[i] = the alpha for nodes with i zeros
        alpha1 = alpha_sigmoid[self.zero1_seq]
        alpha2 = alpha_sigmoid[self.zero2_seq]
        alpha3 = alpha_sigmoid[self.zero3_seq]
        alpha_sigmoid_stack = torch.stack((alpha3, alpha2, alpha1), dim=1)
        
        n_round = self.n_round
        
        # alpha_full: (2^k, )
        # alpha_full = alpha_sigmoid[self.zero_index]
        # Q_full: (2^k, 2^k)
        # Q_full = torch.outer(alpha_full, alpha_full)                       
        
        # q12 = alpha1 * alpha2
        # q13 = alpha1 * alpha3
        # q23 = alpha2 * alpha3
        # Q, R, P_Rem: (n_indices, 3)
        # Q = torch.stack((q12, q13, q23), dim=1)        
        # r_stack = self.R_num / Q
        # r_stack.clamp_(max=1)        
        
        # E0_local = torch.zeros(kr_power + 1).to(device)
        # E1_local = torch.zeros(kr_power + 1).to(device)
        # E2_local = torch.zeros(kr_power + 1).to(device)
        # E3_local = torch.zeros(kr_power + 1).to(device)
        
        E0 = torch.tensor(0.).to(device)
        E1 = torch.tensor(0.).to(device)
        E2 = torch.tensor(0.).to(device)
        E3 = torch.tensor(0.).to(device)
        
        P_stack = self.p_stack        
        P_sorted, indices = torch.sort(P_stack, dim=-1)
        P_min, P_min_indices = P_sorted[..., 0], indices[..., 0]
        P_mid, P_mid_indices = P_sorted[..., 1], indices[..., 1]
        P_max, P_max_indices = P_sorted[..., 2], indices[..., 2]
        
        alpha_max = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_max_indices.unsqueeze(-1)).squeeze(-1)
        alpha_mid = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_mid_indices.unsqueeze(-1)).squeeze(-1)
        alpha_min = torch.gather(alpha_sigmoid_stack, dim=-1, index=P_min_indices.unsqueeze(-1)).squeeze(-1)                
                
        n_indices = self.n_indices
        # sorted each row of r_stack and record the order
        # r_stack_sorted: (n_indices, 3)
        
        pg_3 = alpha_min * alpha_mid * alpha_max
        # the two endpoints of rmin sampled
        pg_2_min = (1 - alpha_min) * alpha_mid * alpha_max
        pg_2_mid = alpha_min * (1 - alpha_mid) * alpha_max
        pg_2_max = alpha_min * alpha_mid * (1 - alpha_max)
        pg_lt2 = 1 - pg_3 - pg_2_min - pg_2_mid - pg_2_max        
                
        # compute the probability that all three pairs are sampled after n_round rounds
        # P_0 = pg_lt2 ** self.n_round  # no pair is sampled
        # P_1_max = ((pg_lt2 + pg_2_max) ** self.n_round) - P_0  # only the pair with P_max is sampled
        # P_1_mid = ((pg_lt2 + pg_2_mid) ** self.n_round) - P_0  # only the pair with P_mid is sampled
        # P_1_min = ((pg_lt2 + pg_2_min) ** self.n_round) - P_0  # only the pair with P_min is sampled
        # P_2_max_mid = ((pg_lt2 + pg_2_max + pg_2_mid) ** self.n_round) - P_1_max - P_1_mid - P_0  # the pairs with P_max and P_mid are sampled
        # P_2_max_min = ((pg_lt2 + pg_2_max + pg_2_min) ** self.n_round) - P_1_max - P_1_min - P_0  # the pairs with P_max and P_min are sampled
        # P_2_mid_min = ((pg_lt2 + pg_2_mid + pg_2_min) ** self.n_round) - P_1_mid - P_1_min - P_0  # the pairs with P_mid and P_min are sampled
        # P_3 = 1 - P_0 - P_1_max - P_1_mid - P_1_min - P_2_max_mid - P_2_max_min - P_2_mid_min  # all three pairs are sampled
        
        # conditioned on all three pairs are sampled, compute the probability that the motif is sampled
        pg_geq2 = pg_3 + pg_2_min + pg_2_mid + pg_2_max + num_eps
        pg_geq2_ex_max = pg_3 + pg_2_min + pg_2_mid + num_eps
        pg_geq2_ex_mid = pg_3 + pg_2_min + pg_2_max + num_eps
        pg_geq2_ex_min = pg_3 + pg_2_mid + pg_2_max + num_eps
        
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
                
        E0 += (self.n_triplet_seq * p_0).sum()
        E1 += (self.n_triplet_seq * p_1).sum()
        E2 += (self.n_triplet_seq * p_2).sum()
        E3 += (self.n_triplet_seq * p_3).sum()
                
        # n_isolated = isolated_nodes(R_full, Prem_full, alpha_full, self.n_round_int)
        # n_nonisolated = self.n_total - n_isolated
        
        return E0, E1, E2, E3

# inverse sigmoid
def inv_sigmoid(x: float):
    return torch.log(x / (1 - x))

seed_matrix = torch.tensor(seed_matrix_dict[args.dataset]).to(device)
# bound to [0, 1]
seed_matrix.clamp_(min=0, max=1)
seed_matrix = (seed_matrix + seed_matrix.T) / 2


p_data = Path("../data/nx_graph")

with open(p_data / f"{args.dataset}.graph", "rb") as f:
    graph: nx.Graph = pickle.load(f)
n_gt = graph.number_of_nodes()
m_gt = graph.number_of_edges()
deg_list = [len(graph[v]) for v in graph]
n_triangles = sum(nx.triangles(graph).values())
n_wedges = sum(d * (d - 1) / 2 for d in deg_list)
n_open = n_wedges - n_triangles
n_triangles /= 3
ratio_gt = n_triangles / n_open
k = k_dict[args.dataset]

alpha_raw = [args.alpha for _ in range(k + 1)]
alpha_inv_sigmoid = inv_sigmoid(torch.tensor(alpha_raw))

model = iterBindingKronecker(alpha_inv_sigmoid, args.nround, seed_matrix, k).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epsilon = 1e-3
loss_eps = 1e-16
for i in tqdm(range(args.ep), desc="Training", unit="epoch"):
    optimizer.zero_grad()
    E_0, E_1, E_2, E_3 = model()    
    ratio_gen = E_3 / E_2
    loss = (1 - (E_3 * 3 / n_triangles)) ** 2 + args.ww * (1 - (E_2 / n_open)) ** 2
    loss_triangle = (1 - (E_3 / n_triangles)) ** 2
    loss_wedge = (1 - (E_2 / n_open)) ** 2
    loss_ratio = (1 - (ratio_gen / ratio_gt)) ** 2
    # loss_node = (1 - (E_n / n_gt)) ** 2
    loss = args.wt * loss_triangle + args.ww * loss_wedge + args.wr * loss_ratio
    if loss < loss_eps:
        break
    # loss = (E_3 * 3 - n_triangles) ** 2
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)
    for p in model.parameters():
        if torch.isnan(p.grad).any():
            p.grad = torch.where(torch.isnan(p.grad), torch.zeros_like(p.grad), p.grad)
    # grad climping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)
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
seed_matrix = seed_matrix.detach().cpu().numpy()
k = model.k
n_round = model.n_round_int

# mkdir if not exists
p_res = f"../fit_res/KR_iter/{args.name}"
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

