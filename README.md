# Edge Probability Graph Models Beyond Edge Independency

This repository contains the appendix, code, and data for the paper "Edge Probability Graph Models Beyond Edge Independency" published in the IEEE International Conference on Data Mining (ICDM) 2025.

Paper link (TBD):

- [IEEE Xplore]()
- [arXiv (full version)](https://arxiv.org/abs/2405.16726)

## Data Preparation (Folder `data`)

### Overview

We generate graphs for the original edge-independent graphs models (EIGMs).

### Contents

Files:

- `kron_seed.py`: This file contains the Kronecker seed matrix, generates the Kronecker edge-probability matrix for each dataset, and generates random graphs using the original edge-independent Kronecker (KR) model.
- `er_cl_gen.py`: This file generates random graphs using the original edge-independent Erdos-Renyi (ER) and Chung-Lu (CL) models.
- `sbm_gen.py`: This file generates random graphs using the original edge-independent stochastic block (SB) model.

Subfolders:

- `gt_txt`: The original graphs in the undirected edge list format.
- `gt_txt_directed`: The original graphs in the directed edge list format (i.e., each edge is repeated twice).
- `nx_graph`: The original graphs in the NetworkX format.
- `res_kronfit`: The fitting results for the original Kronecker (KR) model using the [KronFit](https://snap.stanford.edu/snap/description.html).
- `sbm_PB_NB`: The parameters for the stochastic block (SB) model.

### How to run

```bash
cd data
python kron_seed.py
python er_cl_gen.py
python sbm_gen.py
cd ..
```

### Outputs

The code generates the following folder(s):

- `orig_er`: Random graphs generated using the original edge-independent Erdos-Renyi (ER) model.
- `orig_cl`: Random graphs generated using the original edge-independent Chung-Lu (CL) model.
- `orig_sbm`: Random graphs generated using the original edge-independent stochastic block (SB) model.
- `kron_probs`: Kronecker edge-probability matrices.
- `orig_kr`: Random graphs generated using the original edge-independent Kronecker (KR) model.
- Note: The folders are generated inside the `data` folder.

## Fitting (Folder `fitting`)

### Overview

We do parameter fitting for the proposed edge-probability graph models (EPGMs).

### Contents

Files:

- `ER_iid.wls`: The fitting code for the Erdos-Renyi (ER) model with parallel binding.
- `ER_iter.wls`: The fitting code for the Erdos-Renyi (ER) model with local binding.
- `CL_iid.py`: The fitting code for the Chung-Lu (CL) model with parallel binding.
- `CL_iter.py`: The fitting code for the Chung-Lu (CL) model with local binding.
- `SBM_iid.py`: The fitting code for the stochastic block (SB) model with parallel binding.
- `SBM_iter.py`: The fitting code for the stochastic block (SB) model with local binding.
- `KR_iid.py`: The fitting code for the stochastic Kronecker (KR) model with parallel binding.
- `KR_iter.py`: The fitting code for the stochastic Kronecker (KR) model with local binding.
- `fitting.sh`: The script to run the fitting code.

### How to run

```bash
cd fitting
bash fitting.sh all
cd ..
```

### Outputs

The code generates the following folder(s):

- `fit_res`: The fitting results.
- Note: The folder is generated outside the `fitting` folder, i.e., in the root directory.

### Note

- For the Erdos-Renyi (ER) model, we use the Wolfram Language for better numerical accuracy.
- For the other models, we use Python (specifically, gradient descent with PyTorch). We used PyTorch 1.9.0, but it should work with many other versions.

## Generation (Folder `generation`)

### Overview

We generate graphs using the proposed edge-probability graph models (EPGMs).

### Contents

Files:

- `gen_iid_er.cpp`: The graph generation code for the Erdos-Renyi (ER) model with parallel binding.
- `gen_iter_er.cpp`: The graph generation code for the Erdos-Renyi (ER) model with local binding.
- `gen_iid_cl.cpp`: The graph generation code for the Chung-Lu (CL) model with parallel binding.
- `gen_iter_cl.cpp`: The graph generation code for the Chung-Lu (CL) model with local binding.
- `gen_iid_sbm.cpp`: The graph generation code for the stochastic block (SB) model with parallel binding.
- `gen_iter_sbm.cpp`: The graph generation code for the stochastic block (SB) model with local binding.
- `gen_iid_kr.cpp`: The graph generation code for the stochastic Kronecker (KR) model with parallel binding.
- `gen_iter_kr.cpp`: The graph generation code for the stochastic Kronecker (KR) model with local binding.
- `graph_analy.cpp`: The code to analyze the graph statistics of the generated graphs.
- `gen_iid_er.sh`: The script to run the graph generation code for the Erdos-Renyi (ER) model with parallel binding.
- `gen_iter_er.sh`: The script to run the graph generation code for the Erdos-Renyi (ER) model with local binding.
- `gen_iid_cl.sh`: The script to run the graph generation code for the Chung-Lu (CL) model with parallel binding.
- `gen_iter_cl.sh`: The script to run the graph generation code for the Chung-Lu (CL) model with local binding.
- `gen_iid_sbm.sh`: The script to run the graph generation code for the stochastic block (SB) model with parallel binding.
- `gen_iter_sbm.sh`: The script to run the graph generation code for the stochastic block (SB) model with local binding.
- `gen_iid_kr.sh`: The script to run the graph generation code for the stochastic Kronecker (KR) model with parallel binding.
- `gen_iter_kr.sh`: The script to run the graph generation code for the stochastic Kronecker (KR) model with local binding.
- `gen.sh`: The script to run the graph generation code for all the models.
- `compile.sh`: The script to compile the graph generation code.
- `gen_scratch_er_iid.sh`: The script to run the graph generation code for the Erdos-Renyi (ER) model with parallel binding, using edge probabilities "from scratch", i.e., not using results fit to a specific graph but manipulate the parameters directly.
- `gen_scratch_er_iter.sh`: The script to run the graph generation code for the Erdos-Renyi (ER) model with local binding, using edge probabilities "from scratch".
- `gen_scratch_cl_iid.sh`: The script to run the graph generation code for the Chung-Lu (CL) model with parallel binding, using edge probabilities "from scratch".
- `gen_scratch_cl_iter.sh`: The script to run the graph generation code for the Chung-Lu (CL) model with local binding, using edge probabilities "from scratch".
- `gen_scratch_sbm_iid.sh`: The script to run the graph generation code for the stochastic block (SB) model with parallel binding, using edge probabilities "from scratch".
- `gen_scratch_sbm_iter.sh`: The script to run the graph generation code for the stochastic block (SB) model with local binding, using edge probabilities "from scratch".
- `gen_scratch_kr_iid.sh`: The script to run the graph generation code for the stochastic Kronecker (KR) model with parallel binding, using edge probabilities "from scratch".
- `gen_scratch_kr_iter.sh`: The script to run the graph generation code for the stochastic Kronecker (KR) model with local binding, using edge probabilities "from scratch".

Subfolders:

- `optimized`: The graph generation code optimized for large-scale graphs.

### How to run

```bash
cd generation
bash compile.sh
bash gen.sh
```

### Outputs

The code generates the following folder(s):

- `gen_res`: The generation results.
- Note: The folder is generated outside the `generation` folder, i.e., in the root directory.

## Appendix

See the PDF file in the root directory for the full version with the appendix: `full_paper_with_appendix.pdf`.

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{Bu2025EPGM,
  title={Edge Probability Graph Models Beyond Edge Independency: Concepts, Analyses, and Algorithms},
  author={Bu, Fanchen and Yang, Ruochen and Bogdan, Paul and Shin, Kijung},
  booktitle={ICDM},
  year={2025}
}
```
