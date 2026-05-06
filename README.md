# APC-523 Final Project: Experiments 1-4

This repository contains four standalone scripts for eigenvalue experiments:

1. **Experiment 1: Spectral Gap and Power Iteration**  
   `实验1/experiment1_spectral_gap_power_iteration.py`
2. **Experiment 2: Clustered Dominant Eigenvalues and Subspace Iteration**  
   `实验2/experiment2_clustered_subspace_iteration.py`
3. **Experiment 3: Dense Symmetric Matrices (Basic QR vs. Lanczos)**  
   `实验3/experiment3_dense_qr_vs_lanczos.py`
4. **Experiment 4: Sparse Symmetric Laplacian and Lanczos Iteration**  
   `实验4/experiment4_sparse_laplacian_lanczos.py`

## Requirements

- Python 3.9 or newer
- Python packages:
  - `numpy`
  - `scipy`
  - `matplotlib`

Install dependencies with:

```bash
pip install numpy scipy matplotlib
```

## Run Instructions

All four experiment scripts are directly runnable. No extra pipeline is required.

From the repository root, run:

```bash
python 实验1/experiment1_spectral_gap_power_iteration.py
python 实验2/experiment2_clustered_subspace_iteration.py
python 实验3/experiment3_dense_qr_vs_lanczos.py
python 实验4/experiment4_sparse_laplacian_lanczos.py
```

Each script runs independently and writes its own output figures/data files into its corresponding experiment folder.
