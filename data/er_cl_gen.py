import networkx as nx
import numpy as np
from pathlib import Path
import pickle
from tqdm.auto import tqdm, trange

dataset_list = [
    "facebook",
    "hamsterster",
    "email-dnc",
    "polblogs",
    "bio-CE-PG",
    "bio-SC-HT",
]

p_data = Path(".")
p_nx = p_data / "nx_graph"

N_GRAPHS = 100
for ds in dataset_list:
    with open(p_nx / f"{ds}.graph", "rb") as f:
        G = pickle.load(f)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = n_edges / (n_nodes * (n_nodes - 1) / 2)
    nodes = sorted(G.nodes())
    degrees = [len(G[v]) for v in nodes]

    # ER
    p_ER = p_data / "orig_er"
    p_ER.mkdir(exist_ok=True)
    for i_graph in trange(N_GRAPHS, desc="ER"):
        G_ER = nx.gnp_random_graph(n_nodes, density, seed=i_graph)    
        # save the graph as edge list
        with open(p_ER / f"{ds}_{i_graph}.txt", "w") as f:
            # f.write(f"{G_ER.number_of_nodes()} {G_ER.number_of_edges()}\n")
            for edge in G_ER.edges():
                f.write(f"{edge[0]} {edge[1]}\n")

    # Chung-Lu
    p_CL = p_data / "orig_cl"
    p_CL.mkdir(exist_ok=True)
    for i_graph in trange(N_GRAPHS, desc="CL"):
        G_CL = nx.expected_degree_graph(degrees, seed=i_graph)
        # save the graph as edge list
        with open(p_CL / f"{ds}_{i_graph}.txt", "w") as f:
            # f.write(f"{G_CL.number_of_nodes()} {G_CL.number_of_edges()}\n")
            for edge in G_CL.edges():
                f.write(f"{edge[0]} {edge[1]}\n")