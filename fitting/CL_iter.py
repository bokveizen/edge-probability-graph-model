import os
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
class iterBindingChungLu(torch.nn.Module):
    def __init__(self, alpha: torch.tensor, n_round: int, D: torch.tensor, ND: torch.tensor):
        super().__init__()
        # make alpha a tensor of parameters
        self.alpha = torch.nn.Parameter(alpha.to(device))
        self.n_round = torch.tensor(float(n_round)).to(device)
        self.D_sum = (D * ND).sum()
        # self.D = torch.nn.Parameter(D.to(device))
        self.D = D.to(device)
        self.ND = ND.to(device)
        self.n_total = self.ND.sum()
        assert self.D.shape == self.ND.shape == self.alpha.shape, "D, ND, and alpha must have the same shape (the number of degree groups)"
        self.n_deg = self.D.shape[0]
        self.index_mesh = torch.meshgrid(torch.arange(self.n_deg), torch.arange(self.n_deg), torch.arange(self.n_deg))
    
    def forward(self):
        # D = torch.relu(self.D)
        D = self.D
        ND = self.ND
        alpha_sigmoid = torch.sigmoid(self.alpha)        
        n_round = self.n_round
        n_deg = self.n_deg
                
        i, j, k = self.index_mesh
        # compute P: P(d1, d2) = (d1 * d2) / (2 * D.sum())
        P = torch.outer(D, D) / (D * ND).sum()
        P.clamp_(min=0, max=1)
        
        # P_stack: (n_deg, n_deg, n_deg, 3); P_stack[i, j, k] = [P[i, j], P[i, k], P[j, k]]
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
        n_triplets = torch.zeros((n_deg, n_deg, n_deg)).to(device)

        # Condition: i, j, and k are all distinct
        mask = (i != j) & (j != k) & (i != k)
        n_triplets[mask] = ND[i[mask]] * ND[j[mask]] * ND[k[mask]] / 6

        # Condition: i == j != k
        mask = (i == j) & (i != k)
        n_triplets[mask] = ND[i[mask]] * (ND[i[mask]] - 1) * ND[k[mask]] / 6

        # Condition: i == k != j
        mask = (i == k) & (i != j)
        n_triplets[mask] = ND[i[mask]] * (ND[i[mask]] - 1) * ND[j[mask]] / 6

        # Condition: j == k != i
        mask = (j == k) & (j != i)
        n_triplets[mask] = ND[j[mask]] * (ND[j[mask]] - 1) * ND[i[mask]] / 6

        # Condition: i == j == k
        mask = (i == j) & (j == k)
        n_triplets[mask] = ND[i[mask]] * (ND[i[mask]] - 1) * (ND[i[mask]] - 2) / 6
        
        # compute the expected number of 3-motifs
        E_0 = (p_0 * n_triplets).sum()
        E_1 = (p_1 * n_triplets).sum()
        E_2 = (p_2 * n_triplets).sum()
        E_3 = (p_3 * n_triplets).sum()
        
        return E_0, E_1, E_2, E_3, (D * ND).sum()

p_data = Path("../data/nx_graph")

with open(p_data / f"{args.dataset}.graph", "rb") as f:
    graph: nx.Graph = pickle.load(f)

# inverse sigmoid
def inv_sigmoid(x: float):
    return torch.log(x / (1 - x))

n_gt = graph.number_of_nodes()
m_gt = graph.number_of_edges()
deg_list = [len(graph[v]) for v in graph]
deg_cnt = Counter(deg_list)
D_list = sorted(deg_cnt.keys())
ND_list = [deg_cnt[d] for d in D_list]
D_list = [float(d) for d in D_list]
ND_list = [float(nd) for nd in ND_list]
D, ND = torch.tensor(D_list).to(device), torch.tensor(ND_list).to(device)
Em_orig = (D * ND).sum()
n_triangles = sum(nx.triangles(graph).values())
n_wedges = sum(d * (d - 1) / 2 for d in deg_list)
n_open = n_wedges - n_triangles
n_triangles /= 3
ratio_gt = n_triangles / n_open
n_deg = D.shape[0]


alpha_raw = [args.alpha for _ in range(n_deg)]
alpha_inv_sigmoid = inv_sigmoid(torch.tensor(alpha_raw))

print(alpha_inv_sigmoid.shape, D.shape, ND.shape)
model = iterBindingChungLu(alpha=alpha_inv_sigmoid, n_round=args.nround, D=D, ND=ND).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epsilon = 1e-3
loss_eps = 1e-16
for i in tqdm(range(args.ep), desc="Training", unit="epoch"):
    optimizer.zero_grad()
    E_0, E_1, E_2, E_3, E_m = model()
    ratio_gen = E_3 / E_2
    loss_triangle = (1 - (E_3 / n_triangles)) ** 2
    loss_wedge = (1 - (E_2 / n_open)) ** 2
    loss_ratio = (1 - (ratio_gen / ratio_gt)) ** 2        
    loss = args.wt * loss_triangle + args.ww * loss_wedge + args.wr * loss_ratio        
    # break if loss is small enough
    if loss < loss_eps:
        break    
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
    # grad clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e3)    
    optimizer.step()    
    with torch.no_grad():
        # model.D.clamp_(min=epsilon)
        model.alpha.clamp_(min=-1e3, max=1e3)
    tqdm.write(f"epoch {i} || loss: {loss.item():.4f}, "
                f"E_2: {E_2.item():.4f} -> {n_open}, "
                f"E_3: {E_3.item():.4f} -> {n_triangles}, "
                f"E_m: {E_m.item():.4f} -> {m_gt}, "                
                f"alpha mean: {torch.sigmoid(model.alpha).mean().item():.4f}")

# save the results
alpha = torch.sigmoid(model.alpha).detach().cpu().numpy()
D = D.detach().cpu().numpy()
ND = ND.detach().cpu().numpy()

# mkdir if not exists
p_res = f"../fit_res/CL_iter/{args.name}"
p_res = Path(p_res)
p_res.mkdir(exist_ok=True, parents=True)

with open(f"{p_res}/{args.dataset}.alpha_D_ND", "wb") as f:
    pickle.dump((alpha, D, ND), f)

# and save in a text file
with open(f"{p_res}/{args.dataset}.txt", "w") as f:
    f.write(f"{int(args.nround)}\n")
    for i in range(n_deg):
        f.write(f"{int(D[i])} {int(ND[i])} {alpha[i]:.16f} \n")
