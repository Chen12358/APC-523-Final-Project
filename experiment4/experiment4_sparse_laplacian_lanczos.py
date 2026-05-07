#!/usr/bin/env python3
"""
Experiment 4: Sparse Symmetric Laplacian and Lanczos Iteration

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal


def make_1d_laplacian(n):
    """Construct 1D discrete Laplacian (CSR): tridiag(-1, 2, -1)."""
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    return diags([off, main, off], offsets=[-1, 0, 1], format="csr")


def compute_residual(A, theta, u):
    """Normalized residual ||A u - theta u||_2 / ||u||_2."""
    r = A @ u - theta * u
    return np.linalg.norm(r, 2) / np.linalg.norm(u, 2)


def power_iteration_sparse(A, lambda_ref, max_iter=100, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x /= np.linalg.norm(x)

    errors = np.zeros(max_iter)
    residuals = np.zeros(max_iter)

    for k in range(max_iter):
        y = A @ x
        x = y / np.linalg.norm(y)
        lam = float(x @ (A @ x) / (x @ x))

        errors[k] = abs(lam - lambda_ref)
        residuals[k] = compute_residual(A, lam, x)

    return errors, residuals


def subspace_iteration_sparse(A, top5_ref, max_iter=80, p=5, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    X0 = rng.standard_normal((n, p))
    X, _ = np.linalg.qr(X0)

    largest_errors = np.zeros(max_iter)
    largest_residuals = np.zeros(max_iter)
    top5_max_errors = np.zeros(max_iter)
    top5_max_residuals = np.zeros(max_iter)

    for k in range(max_iter):
        Y = A @ X
        X, _ = np.linalg.qr(Y)

        AX = A @ X
        B = X.T @ AX

        theta, Yb = np.linalg.eigh(B)
        idx = np.argsort(theta)[::-1]
        theta = theta[idx]
        Yb = Yb[:, idx]

        theta5 = theta[:p]
        largest_errors[k] = abs(theta5[0] - top5_ref[0])
        top5_max_errors[k] = np.max(np.abs(theta5 - top5_ref))

        # residuals for Ritz pairs u_j = X y_j
        res = np.zeros(p)
        for j in range(p):
            uj = X @ Yb[:, j]
            res[j] = compute_residual(A, theta5[j], uj)

        largest_residuals[k] = res[0]
        top5_max_residuals[k] = np.max(res)

    return largest_errors, largest_residuals, top5_max_errors, top5_max_residuals


def lanczos_sparse(A, top5_ref, max_iter=80, seed=0):
    """Lanczos with full reorthogonalization (stabilized implementation)."""
    n = A.shape[0]
    m = min(max_iter, n)
    rng = np.random.default_rng(seed)

    Q = np.zeros((n, m), dtype=float)
    alpha = np.zeros(m, dtype=float)
    beta = np.zeros(max(m - 1, 1), dtype=float)

    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)
    q_prev = np.zeros(n, dtype=float)
    beta_prev = 0.0

    largest_errors = np.full(m, np.nan)
    largest_residuals = np.full(m, np.nan)
    top5_errors = np.full(m, np.nan)
    top5_residuals = np.full(m, np.nan)

    k_actual = m

    for j in range(m):
        z = A @ q
        if j > 0:
            z = z - beta_prev * q_prev

        alpha_j = np.dot(q, z)
        z = z - alpha_j * q

        # full reorthogonalization for finite-precision stability
        if j > 0:
            z = z - Q[:, :j] @ (Q[:, :j].T @ z)

        beta_j = np.linalg.norm(z)

        Q[:, j] = q
        alpha[j] = alpha_j
        if j < m - 1:
            beta[j] = beta_j

        # Ritz data from T_k
        k = j + 1
        d = alpha[:k]
        e = beta[:k - 1] if k > 1 else np.array([], dtype=float)

        theta, Yt = eigh_tridiagonal(d, e)
        idx = np.argsort(theta)[::-1]
        theta = theta[idx]
        Yt = Yt[:, idx]

        # largest Ritz pair
        theta1 = theta[0]
        u1 = Q[:, :k] @ Yt[:, 0]
        largest_errors[j] = abs(theta1 - top5_ref[0])
        largest_residuals[j] = compute_residual(A, theta1, u1)

        # top-5 (or fewer in early iterations)
        p_eff = min(5, k)
        theta_p = theta[:p_eff]
        top5_errors[j] = np.max(np.abs(theta_p - top5_ref[:p_eff]))

        res_p = np.zeros(p_eff)
        for t in range(p_eff):
            ut = Q[:, :k] @ Yt[:, t]
            res_p[t] = compute_residual(A, theta_p[t], ut)
        top5_residuals[j] = np.max(res_p)

        # happy breakdown
        if beta_j < 1e-14:
            k_actual = k
            break

        q_prev = q
        q = z / beta_j
        beta_prev = beta_j

    return {
        "largest_errors": largest_errors[:k_actual],
        "largest_residuals": largest_residuals[:k_actual],
        "top5_errors": top5_errors[:k_actual],
        "top5_residuals": top5_residuals[:k_actual],
        "k_actual": k_actual,
    }


def run_experiment():
    n_list = [500, 1000, 2000, 5000]
    power_iter = 100
    subspace_iter = 80
    lanczos_iter = 80
    p = 5

    runtimes = {"power": [], "subspace": [], "lanczos": []}
    final_errors = {"subspace_top5": [], "lanczos_top5": []}

    representative_n = 2000 if 2000 in n_list else n_list[len(n_list) // 2]
    rep_data = {}

    print("Lanczos note: full reorthogonalization enabled for numerical stabilization.")

    for i, n in enumerate(n_list):
        A = make_1d_laplacian(n)

        lam1_ref = eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
        top5_ref = eigsh(A, k=5, which="LA", return_eigenvectors=False)
        top5_ref = np.sort(top5_ref)[::-1]

        t0 = time.time()
        p_err, p_res = power_iteration_sparse(A, lam1_ref, max_iter=power_iter, seed=10 + i)
        t1 = time.time()
        runtimes["power"].append(t1 - t0)

        t0 = time.time()
        s_largest_err, s_largest_res, s_top5_err, s_top5_res = subspace_iteration_sparse(
            A, top5_ref, max_iter=subspace_iter, p=p, seed=20 + i
        )
        t1 = time.time()
        runtimes["subspace"].append(t1 - t0)

        t0 = time.time()
        l_res = lanczos_sparse(A, top5_ref, max_iter=lanczos_iter, seed=30 + i)
        t1 = time.time()
        runtimes["lanczos"].append(t1 - t0)

        final_errors["subspace_top5"].append(s_top5_err[-1])
        final_errors["lanczos_top5"].append(l_res["top5_errors"][-1])

        if n == representative_n:
            rep_data = {
                "n": n,
                "power_errors": p_err,
                "power_residuals": p_res,
                "subspace_largest_errors": s_largest_err,
                "subspace_largest_residuals": s_largest_res,
                "subspace_top5_errors": s_top5_err,
                "subspace_top5_residuals": s_top5_res,
                "lanczos_largest_errors": l_res["largest_errors"],
                "lanczos_largest_residuals": l_res["largest_residuals"],
                "lanczos_top5_errors": l_res["top5_errors"],
                "lanczos_top5_residuals": l_res["top5_residuals"],
            }

        print(
            f"n={n:5d} | power {runtimes['power'][-1]:.3f}s | "
            f"subspace {runtimes['subspace'][-1]:.3f}s | lanczos {runtimes['lanczos'][-1]:.3f}s"
        )

    return {
        "n_list": np.array(n_list, dtype=int),
        "representative_n": int(rep_data["n"]),
        "power_errors": rep_data["power_errors"],
        "power_residuals": rep_data["power_residuals"],
        "subspace_largest_errors": rep_data["subspace_largest_errors"],
        "subspace_largest_residuals": rep_data["subspace_largest_residuals"],
        "subspace_top5_errors": rep_data["subspace_top5_errors"],
        "subspace_top5_residuals": rep_data["subspace_top5_residuals"],
        "lanczos_largest_errors": rep_data["lanczos_largest_errors"],
        "lanczos_largest_residuals": rep_data["lanczos_largest_residuals"],
        "lanczos_top5_errors": rep_data["lanczos_top5_errors"],
        "lanczos_top5_residuals": rep_data["lanczos_top5_residuals"],
        "runtimes_power": np.array(runtimes["power"]),
        "runtimes_subspace": np.array(runtimes["subspace"]),
        "runtimes_lanczos": np.array(runtimes["lanczos"]),
        "final_subspace_top5_errors": np.array(final_errors["subspace_top5"]),
        "final_lanczos_top5_errors": np.array(final_errors["lanczos_top5"]),
    }


def plot_results(data):
    n_list = data["n_list"]
    it_p = np.arange(1, len(data["power_errors"]) + 1)
    it_s = np.arange(1, len(data["subspace_largest_errors"]) + 1)
    it_l = np.arange(1, len(data["lanczos_largest_errors"]) + 1)

    # Figure 1: largest eigenvalue error
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(it_p, data["power_errors"], linewidth=2.0, label="Power iteration")
    plt.semilogy(it_s, data["subspace_largest_errors"], linewidth=2.0, label="Subspace, largest Ritz")
    plt.semilogy(it_l, data["lanczos_largest_errors"], linewidth=2.0, label="Lanczos, largest Ritz")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel(r"$|\hat\lambda-\lambda_{\max}|$", fontsize=12)
    plt.title(f"Largest Eigenvalue Error (n={data['representative_n']})", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("laplacian_largest_error.png")
    plt.savefig("laplacian_largest_error.pdf")
    plt.close()

    # Figure 2: largest eigenpair residual (fair 1-vs-1 comparison)
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(it_p, data["power_residuals"], linewidth=2.0, label="Power, largest eigenpair")
    plt.semilogy(it_s, data["subspace_largest_residuals"], linewidth=2.0, label="Subspace, largest Ritz pair")
    plt.semilogy(it_l, data["lanczos_largest_residuals"], linewidth=2.0, label="Lanczos, largest Ritz pair")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("normalized residual", fontsize=12)
    plt.title(f"Largest Eigenpair Residual (n={data['representative_n']})", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("laplacian_largest_residual.png")
    plt.savefig("laplacian_largest_residual.pdf")
    plt.close()

    # Figure 3: top-5 residual (subspace vs Lanczos only)
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.semilogy(it_s, data["subspace_top5_residuals"], linewidth=2.0, label="Subspace, max residual top-5")
    plt.semilogy(it_l, data["lanczos_top5_residuals"], linewidth=2.0, label="Lanczos, max residual top-5")
    plt.xlabel("iteration", fontsize=12)
    plt.ylabel("normalized residual", fontsize=12)
    plt.title(f"Top-5 Ritz Pair Residuals (n={data['representative_n']})", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("laplacian_top5_residual.png")
    plt.savefig("laplacian_top5_residual.pdf")
    plt.close()

    # Figure 4: runtime vs n (fixed iteration budgets)
    plt.figure(figsize=(7.2, 5.0), dpi=140)
    plt.plot(n_list, data["runtimes_power"], marker="o", linewidth=2.0, label="Power iteration (100)")
    plt.plot(n_list, data["runtimes_subspace"], marker="s", linewidth=2.0, label="Subspace iteration (80)")
    plt.plot(n_list, data["runtimes_lanczos"], marker="^", linewidth=2.0, label="Lanczos (80)")
    plt.xlabel("n", fontsize=12)
    plt.ylabel("runtime (seconds)", fontsize=12)
    plt.title("Runtime vs Matrix Size (Fixed Iteration Counts)", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    plt.tight_layout()
    plt.savefig("laplacian_runtime.png")
    plt.savefig("laplacian_runtime.pdf")
    plt.close()


def main():
    data = run_experiment()
    plot_results(data)

    np.savez(
        "laplacian_lanczos_data.npz",
        n_list=data["n_list"],
        representative_n=data["representative_n"],
        power_errors=data["power_errors"],
        power_residuals=data["power_residuals"],
        subspace_largest_errors=data["subspace_largest_errors"],
        subspace_largest_residuals=data["subspace_largest_residuals"],
        subspace_top5_errors=data["subspace_top5_errors"],
        subspace_top5_residuals=data["subspace_top5_residuals"],
        lanczos_largest_errors=data["lanczos_largest_errors"],
        lanczos_largest_residuals=data["lanczos_largest_residuals"],
        lanczos_top5_errors=data["lanczos_top5_errors"],
        lanczos_top5_residuals=data["lanczos_top5_residuals"],
        runtimes=np.vstack([data["runtimes_power"], data["runtimes_subspace"], data["runtimes_lanczos"]]),
        final_errors=np.vstack([data["final_subspace_top5_errors"], data["final_lanczos_top5_errors"]]),
    )

    print("Done. Generated:")
    print("  laplacian_largest_error.png/.pdf")
    print("  laplacian_largest_residual.png/.pdf")
    print("  laplacian_top5_residual.png/.pdf")
    print("  laplacian_runtime.png/.pdf")
    print("  laplacian_lanczos_data.npz")


if __name__ == "__main__":
    main()
