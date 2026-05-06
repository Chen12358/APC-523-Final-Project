#!/usr/bin/env python3
"""
Experiment 1: Spectral Gap and Power Iteration

Goal:
Study how spectral gap affects power-iteration convergence, and verify that
convergence speed is controlled by |lambda_2 / lambda_1|.
"""

import numpy as np
import matplotlib.pyplot as plt


def make_diagonal_matrix(n: int, lambda1: float, lambda2: float) -> np.ndarray:
    """Construct A = diag(lambda_1, lambda_2, ..., lambda_n).

    lambda_3 ... lambda_n are strictly smaller than lambda_2.
    """
    if n < 3:
        raise ValueError("n must be at least 3")

    # Keep the tail simple and strictly below lambda2 in all test cases.
    # Tail_max is <= 0.4 and always < lambda2 (for lambda2 in {0.5, 0.9, 0.99}).
    tail_max = min(0.4, 0.9 * lambda2)
    tail_min = min(0.01, 0.1 * tail_max)

    tail = np.linspace(tail_max, tail_min, n - 2)

    lambdas = np.empty(n, dtype=float)
    lambdas[0] = lambda1
    lambdas[1] = lambda2
    lambdas[2:] = tail

    # Safety check: all remaining eigenvalues must be strictly < lambda2.
    if not np.all(lambdas[2:] < lambda2):
        raise ValueError("Construction failed: some lambda_i (i>=3) are not < lambda2")

    return np.diag(lambdas)


def power_iteration(A: np.ndarray, x0: np.ndarray, max_iter: int = 80):
    """Run power iteration and record error/residual per iteration."""
    x = x0.copy()

    eig_errors = []
    residuals = []
    rayleighs = []

    for _ in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)

        # Rayleigh quotient
        lambda_k = float(x.T @ (A @ x) / (x.T @ x))

        # Metrics
        eig_err = abs(lambda_k - 1.0)
        res = np.linalg.norm(A @ x - lambda_k * x, 2) / np.linalg.norm(x, 2)

        eig_errors.append(eig_err)
        residuals.append(res)
        rayleighs.append(lambda_k)

    return {
        "eig_error": np.array(eig_errors),
        "residual": np.array(residuals),
        "rayleigh": np.array(rayleighs),
    }


def plot_curves(iter_axis, results, metric_key, ylabel, title, out_png, out_pdf):
    """Plot semilogy curves for three lambda2 settings."""
    plt.figure(figsize=(7.2, 5.0), dpi=140)

    for lambda2, r in results.items():
        ratio = lambda2 / 1.0
        plt.semilogy(
            iter_axis,
            r[metric_key],
            linewidth=2.0,
            label=fr"$\lambda_2={lambda2:.2f}$ (|$\lambda_2/\lambda_1$|={ratio:.2f})",
        )

    plt.xlabel("iteration", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()


def main():
    # Problem setup
    n = 500
    lambda1 = 1.0
    lambda2_values = [0.5, 0.9, 0.99]
    max_iter = 80

    # Same normalized random x0 for fair comparison
    np.random.seed(0)
    x0 = np.random.randn(n)
    x0 = x0 / np.linalg.norm(x0)

    # Run experiments
    results = {}
    for lambda2 in lambda2_values:
        A = make_diagonal_matrix(n=n, lambda1=lambda1, lambda2=lambda2)
        results[lambda2] = power_iteration(A, x0, max_iter=max_iter)

    # Save data for report/repro
    np.savez(
        "power_gap_data.npz",
        n=n,
        lambda1=lambda1,
        lambda2_values=np.array(lambda2_values),
        max_iter=max_iter,
        eig_error_l2_050=results[0.5]["eig_error"],
        eig_error_l2_090=results[0.9]["eig_error"],
        eig_error_l2_099=results[0.99]["eig_error"],
        residual_l2_050=results[0.5]["residual"],
        residual_l2_090=results[0.9]["residual"],
        residual_l2_099=results[0.99]["residual"],
    )

    iters = np.arange(1, max_iter + 1)

    # Figure 1: eigenvalue error
    plot_curves(
        iter_axis=iters,
        results=results,
        metric_key="eig_error",
        ylabel=r"$|\lambda_k - \lambda_1|$",
        title="Convergence of Power Iteration for Different Spectral Gaps",
        out_png="power_gap_error.png",
        out_pdf="power_gap_error.pdf",
    )

    # Figure 2: residual norm
    plot_curves(
        iter_axis=iters,
        results=results,
        metric_key="residual",
        ylabel=r"$\|Ax-\lambda x\|_2 / \|x\|_2$",
        title="Residual Norm of Power Iteration for Different Spectral Gaps",
        out_png="power_gap_residual.png",
        out_pdf="power_gap_residual.pdf",
    )

    print("Done. Generated:")
    print("  power_gap_error.png / power_gap_error.pdf")
    print("  power_gap_residual.png / power_gap_residual.pdf")
    print("  power_gap_data.npz")


if __name__ == "__main__":
    main()
