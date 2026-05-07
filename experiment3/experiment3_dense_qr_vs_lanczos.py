#!/usr/bin/env python3
"""
Experiment 3: Dense Symmetric Matrices and the QR Algorithm

Compare basic QR algorithm (full-spectrum, dense) vs Lanczos iteration
(extremal-eigenvalue oriented) on dense symmetric random matrices.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


def make_dense_symmetric_matrix(n, seed=0):
    """A = (G + G^T)/(2*sqrt(n)), with G standard normal."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n))
    A = (G + G.T) / (2.0 * np.sqrt(n))
    return A


def basic_qr_algorithm(A, max_iter=300):
    """Basic unshifted QR iteration for dense symmetric matrices."""
    Ak = A.copy()
    offdiag_history = []

    for _ in range(max_iter):
        Q, R = np.linalg.qr(Ak)
        Ak = R @ Q

        d = np.diag(np.diag(Ak))
        offdiag_norm = np.linalg.norm(Ak - d, ord='fro')
        offdiag_history.append(offdiag_norm)

    approx_eigs = np.diag(Ak).copy()
    return {
        "approx_eigs": approx_eigs,
        "offdiag_history": np.array(offdiag_history),
        "Ak_final": Ak,
    }


def lanczos_iteration(A, m, rng):
    """m-step Lanczos factorization A Q_m = Q_m T_m + beta_m q_{m+1} e_m^T."""
    n = A.shape[0]
    Q = np.zeros((n, m), dtype=float)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(max(m - 1, 1), dtype=float)

    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)
    q_prev = np.zeros(n, dtype=float)
    beta_prev = 0.0

    k_actual = m

    for j in range(m):
        z = A @ q
        if j > 0:
            z = z - beta_prev * q_prev

        alpha_j = np.dot(q, z)
        z = z - alpha_j * q

        # full re-orthogonalization for numerical stability
        if j > 0:
            z = z - Q[:, :j] @ (Q[:, :j].T @ z)

        beta_j = np.linalg.norm(z)

        Q[:, j] = q
        alpha[j] = alpha_j

        if j < m - 1:
            beta[j] = beta_j

        if beta_j < 1e-14:
            k_actual = j + 1
            break

        q_prev = q
        q = z / beta_j
        beta_prev = beta_j

    alpha = alpha[:k_actual]
    if k_actual > 1:
        beta_used = beta[:k_actual - 1]
    else:
        beta_used = np.array([], dtype=float)

    return Q[:, :k_actual], alpha, beta_used


def compute_lanczos_history(A, Qm, alpha, beta, lambda_max_true):
    """Largest Ritz value error/residual history from T_k, k=1..m."""
    m = len(alpha)
    largest_errors = np.zeros(m)
    residual_history = np.zeros(m)

    for k in range(1, m + 1):
        d = alpha[:k]
        e = beta[:k - 1] if k > 1 else np.array([], dtype=float)

        theta, Y = eigh_tridiagonal(d, e)
        idx = np.argsort(theta)[::-1]
        theta = theta[idx]
        Y = Y[:, idx]

        theta_max = theta[0]
        y_max = Y[:, 0]

        # Ritz vector u = Q_k y
        u = Qm[:, :k] @ y_max
        res = np.linalg.norm(A @ u - theta_max * u, 2) / np.linalg.norm(u, 2)

        largest_errors[k - 1] = abs(theta_max - lambda_max_true)
        residual_history[k - 1] = res

    return largest_errors, residual_history


def run_experiment():
    n_list = [100, 200, 400]
    max_iter_qr = 300

    qr_runtimes = []
    lanczos_runtimes = []
    qr_full_errors = []
    qr_final_offdiag = []
    lanczos_largest_final_errors = []
    lanczos_largest_final_residuals = []

    qr_offdiag_histories = []
    lanczos_largest_errors = []
    lanczos_residual_histories = []

    for i, n in enumerate(n_list):
        A = make_dense_symmetric_matrix(n=n, seed=100 + i)

        # Reference full spectrum
        w_ref = np.linalg.eigh(A)[0]
        w_ref = np.sort(w_ref)[::-1]
        lambda_max_true = w_ref[0]

        # QR algorithm
        t0 = time.time()
        qr_res = basic_qr_algorithm(A, max_iter=max_iter_qr)
        t1 = time.time()

        qr_runtime = t1 - t0
        approx_qr = np.sort(qr_res["approx_eigs"])[::-1]
        qr_err = np.max(np.abs(approx_qr - w_ref))

        qr_runtimes.append(qr_runtime)
        qr_full_errors.append(qr_err)
        qr_final_offdiag.append(qr_res["offdiag_history"][-1])
        qr_offdiag_histories.append(qr_res["offdiag_history"])

        # Lanczos
        m = min(60, n)
        rng = np.random.default_rng(500 + i)

        t0 = time.time()
        Qm, alpha, beta = lanczos_iteration(A, m=m, rng=rng)
        largest_err_hist, residual_hist = compute_lanczos_history(
            A, Qm, alpha, beta, lambda_max_true
        )
        t1 = time.time()

        lanczos_runtime = t1 - t0
        lanczos_runtimes.append(lanczos_runtime)
        lanczos_largest_final_errors.append(largest_err_hist[-1])
        lanczos_largest_final_residuals.append(residual_hist[-1])

        lanczos_largest_errors.append(largest_err_hist)
        lanczos_residual_histories.append(residual_hist)

        print(
            f"n={n:4d} | QR: time={qr_runtime:.3f}s, full_err={qr_err:.3e}, "
            f"offdiag={qr_res['offdiag_history'][-1]:.3e} | "
            f"Lanczos: time={lanczos_runtime:.3f}s, largest_err={largest_err_hist[-1]:.3e}, "
            f"res={residual_hist[-1]:.3e}"
        )

    results = {
        "n_list": np.array(n_list, dtype=int),
        "qr_runtimes": np.array(qr_runtimes),
        "lanczos_runtimes": np.array(lanczos_runtimes),
        "qr_full_errors": np.array(qr_full_errors),
        "qr_final_offdiag": np.array(qr_final_offdiag),
        "lanczos_largest_final_errors": np.array(lanczos_largest_final_errors),
        "lanczos_largest_final_residuals": np.array(lanczos_largest_final_residuals),
        "qr_offdiag_histories": np.array(qr_offdiag_histories, dtype=object),
        "lanczos_largest_errors": np.array(lanczos_largest_errors, dtype=object),
        "lanczos_residual_histories": np.array(lanczos_residual_histories, dtype=object),
    }
    return results


def plot_results(results):
    n_list = results["n_list"]

    # Figure 1: QR off-diagonal norm vs iteration
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    for n, hist in zip(n_list, results["qr_offdiag_histories"]):
        it = np.arange(1, len(hist) + 1)
        plt.semilogy(it, hist, linewidth=2.0, label=f"n={n}")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel(r"$\|A_k-\mathrm{diag}(\mathrm{diag}(A_k))\|_F$", fontsize=12)
    plt.title("Convergence of Basic QR Algorithm on Dense Symmetric Matrices", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("dense_qr_offdiag.png")
    plt.savefig("dense_qr_offdiag.pdf")
    plt.close()

    # Figure 2: runtime vs matrix size
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.plot(n_list, results["qr_runtimes"], marker="o", linewidth=2.0, label="Basic QR")
    plt.plot(n_list, results["lanczos_runtimes"], marker="s", linewidth=2.0, label="Lanczos")
    plt.xlabel("n", fontsize=12)
    plt.ylabel("runtime (seconds)", fontsize=12)
    plt.title("Runtime Comparison on Dense Symmetric Matrices", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("dense_qr_lanczos_runtime.png")
    plt.savefig("dense_qr_lanczos_runtime.pdf")
    plt.close()

    # Figure 3: Lanczos largest eigenvalue error vs iteration
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    for n, err_hist in zip(n_list, results["lanczos_largest_errors"]):
        it = np.arange(1, len(err_hist) + 1)
        plt.semilogy(it, err_hist, linewidth=2.0, label=f"n={n}")
    plt.xlabel("Lanczos iteration m", fontsize=12)
    plt.ylabel(r"$|\theta_m-\lambda_{\max}|$", fontsize=12)
    plt.title("Lanczos Convergence for Largest Eigenvalue", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("dense_lanczos_largest_error.png")
    plt.savefig("dense_lanczos_largest_error.pdf")
    plt.close()

    # Figure 4: summary of final errors
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(n_list, results["qr_full_errors"], marker="o", linewidth=2.0, label="QR full-spectrum max error")
    plt.semilogy(
        n_list,
        results["lanczos_largest_final_errors"],
        marker="s",
        linewidth=2.0,
        label="Lanczos largest-eigenvalue error",
    )
    plt.xlabel("n", fontsize=12)
    plt.ylabel("final error", fontsize=12)
    plt.title("Final Error Summary", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("dense_final_errors.png")
    plt.savefig("dense_final_errors.pdf")
    plt.close()


def main():
    results = run_experiment()
    plot_results(results)

    np.savez(
        "dense_qr_lanczos_data.npz",
        n_list=results["n_list"],
        qr_runtimes=results["qr_runtimes"],
        lanczos_runtimes=results["lanczos_runtimes"],
        qr_full_errors=results["qr_full_errors"],
        qr_offdiag_histories=results["qr_offdiag_histories"],
        lanczos_largest_errors=results["lanczos_largest_errors"],
        lanczos_residual_histories=results["lanczos_residual_histories"],
    )

    print("Done. Generated:")
    print("  dense_qr_offdiag.png/.pdf")
    print("  dense_qr_lanczos_runtime.png/.pdf")
    print("  dense_lanczos_largest_error.png/.pdf")
    print("  dense_final_errors.png/.pdf")
    print("  dense_qr_lanczos_data.npz")


if __name__ == "__main__":
    main()
