#!/usr/bin/env python3
"""
Experiment 2: Clustered Dominant Eigenvalues and Subspace Iteration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr


def make_clustered_symmetric_matrix(n=300, seed=0):
    """Construct dense symmetric A = Q Lambda Q^T with clustered leading eigenvalues."""
    np.random.seed(seed)

    # Random orthogonal matrix Q
    G = np.random.randn(n, n)
    Q, _ = qr(G, mode="economic")

    # Eigenvalues
    top5 = np.array([1.00, 0.99, 0.98, 0.97, 0.96], dtype=float)
    tail = np.linspace(0.5, 0.01, n - 5)
    lambdas = np.concatenate([top5, tail])

    A = Q @ np.diag(lambdas) @ Q.T
    A = 0.5 * (A + A.T)  # enforce symmetry numerically
    return A, lambdas


def compute_ritz_residuals(A, X, thetas, Y):
    """Compute residual norms for Ritz pairs (theta_j, u_j), u_j = X @ y_j."""
    p = len(thetas)
    residuals = np.zeros(p)

    for j in range(p):
        uj = X @ Y[:, j]
        rj = A @ uj - thetas[j] * uj
        residuals[j] = np.linalg.norm(rj, 2) / np.linalg.norm(uj, 2)

    return residuals


def subspace_iteration(A, p, max_iter, true_top5, seed=0):
    """Subspace iteration to approximate leading p eigenpairs."""
    n = A.shape[0]

    np.random.seed(seed)
    X0 = np.random.randn(n, p)
    X, _ = np.linalg.qr(X0)

    top5_values_history = np.zeros((max_iter, p))
    top5_max_errors = np.zeros(max_iter)
    top5_max_residuals = np.zeros(max_iter)

    for k in range(max_iter):
        Y = A @ X
        X, _ = np.linalg.qr(Y)

        B = X.T @ A @ X
        theta, eigvecs_B = np.linalg.eigh(B)

        # sort descending
        idx = np.argsort(theta)[::-1]
        theta = theta[idx]
        eigvecs_B = eigvecs_B[:, idx]

        top5_values_history[k, :] = theta[:p]

        # max abs error over top-5 eigenvalues
        top5_max_errors[k] = np.max(np.abs(theta[:p] - true_top5))

        # max residual over top-5 Ritz pairs
        residuals = compute_ritz_residuals(A, X, theta[:p], eigvecs_B[:, :p])
        top5_max_residuals[k] = np.max(residuals)

    return {
        "top5_values_history": top5_values_history,
        "top5_max_errors": top5_max_errors,
        "top5_max_residuals": top5_max_residuals,
    }


def power_iteration(A, max_iter, lambda1, seed=0):
    """Power iteration for largest eigenvalue only."""
    n = A.shape[0]

    np.random.seed(seed)
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    largest_errors = np.zeros(max_iter)
    residuals = np.zeros(max_iter)
    lambda_history = np.zeros(max_iter)

    for k in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)

        lam = float(x.T @ (A @ x) / (x.T @ x))
        lambda_history[k] = lam
        largest_errors[k] = abs(lam - lambda1)
        residuals[k] = np.linalg.norm(A @ x - lam * x, 2) / np.linalg.norm(x, 2)

    return {
        "largest_errors": largest_errors,
        "residuals": residuals,
        "lambda_history": lambda_history,
    }


def plot_results(
    iterations,
    subspace_top5_errors,
    subspace_top5_residuals,
    subspace_top5_values_history,
    power_largest_errors,
    true_top5,
):
    """Generate all required plots and save to png/pdf."""

    # Figure 1: subspace top-5 max error
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(iterations, subspace_top5_errors, linewidth=2.0, color="tab:blue")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel(r"$\max_{j=1,\ldots,5}|\hat\lambda_j-\lambda_j|$", fontsize=12)
    plt.title("Subspace Iteration for Clustered Leading Eigenvalues", fontsize=13)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("subspace_cluster_error.png")
    plt.savefig("subspace_cluster_error.pdf")
    plt.close()

    # Figure 2: subspace top-5 max residual
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(iterations, subspace_top5_residuals, linewidth=2.0, color="tab:orange")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("max residual over top 5 Ritz pairs", fontsize=12)
    plt.title("Residual Norms for Subspace Iteration", fontsize=13)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("subspace_cluster_residual.png")
    plt.savefig("subspace_cluster_residual.pdf")
    plt.close()

    # Figure 3: power vs subspace for largest eigenvalue error
    largest_subspace_errors = np.abs(subspace_top5_values_history[:, 0] - true_top5[0])

    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(iterations, power_largest_errors, linewidth=2.0, label="Power iteration")
    plt.semilogy(
        iterations,
        largest_subspace_errors,
        linewidth=2.0,
        label="Subspace iteration, largest Ritz value",
    )
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel(r"$|\hat\lambda-\lambda_1|$", fontsize=12)
    plt.title("Largest Eigenvalue Error: Power vs Subspace", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("power_vs_subspace_largest_error.png")
    plt.savefig("power_vs_subspace_largest_error.pdf")
    plt.close()

    # Figure 4: top-5 approximate eigenvalues over iterations
    plt.figure(figsize=(7.5, 5.2), dpi=140)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for j in range(5):
        plt.plot(
            iterations,
            subspace_top5_values_history[:, j],
            linewidth=2.0,
            color=colors[j],
            label=fr"$\hat\lambda_{j+1}$",
        )
        plt.axhline(
            true_top5[j],
            linestyle="--",
            linewidth=1.2,
            color=colors[j],
            alpha=0.8,
        )

    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("approximate eigenvalue", fontsize=12)
    plt.title("Top-5 Ritz Values in Subspace Iteration", fontsize=13)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("subspace_cluster_values.png")
    plt.savefig("subspace_cluster_values.pdf")
    plt.close()


def main():
    n = 300
    p = 5
    max_iter = 80

    A, _ = make_clustered_symmetric_matrix(n=n, seed=0)

    # Reference eigenvalues (descending)
    w, _ = np.linalg.eigh(A)
    w = np.sort(w)[::-1]
    true_top5 = w[:5]

    # Run subspace iteration
    subspace_res = subspace_iteration(
        A=A,
        p=p,
        max_iter=max_iter,
        true_top5=true_top5,
        seed=0,
    )

    # Run power iteration for comparison
    power_res = power_iteration(
        A=A,
        max_iter=max_iter,
        lambda1=true_top5[0],
        seed=0,
    )

    iterations = np.arange(1, max_iter + 1)

    # Save figures
    plot_results(
        iterations=iterations,
        subspace_top5_errors=subspace_res["top5_max_errors"],
        subspace_top5_residuals=subspace_res["top5_max_residuals"],
        subspace_top5_values_history=subspace_res["top5_values_history"],
        power_largest_errors=power_res["largest_errors"],
        true_top5=true_top5,
    )

    # Save data
    np.savez(
        "subspace_cluster_data.npz",
        iterations=iterations,
        subspace_top5_errors=subspace_res["top5_max_errors"],
        subspace_top5_residuals=subspace_res["top5_max_residuals"],
        subspace_top5_values_history=subspace_res["top5_values_history"],
        power_largest_errors=power_res["largest_errors"],
        power_residuals=power_res["residuals"],
        true_top5=true_top5,
    )

    print("Done. Generated files:")
    print("  subspace_cluster_error.png/.pdf")
    print("  subspace_cluster_residual.png/.pdf")
    print("  power_vs_subspace_largest_error.png/.pdf")
    print("  subspace_cluster_values.png/.pdf")
    print("  subspace_cluster_data.npz")


if __name__ == "__main__":
    main()
