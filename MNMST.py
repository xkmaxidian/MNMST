import numpy as np
from scipy.sparse.linalg import svds
from tqdm import trange
from network import create_sppmi_mtx


def MNMST_representation(network_exp, network_spat, lamb=10, gamma=10, dim=150):
    # Variables init
    W1 = create_sppmi_mtx(network_exp, 1)
    W2 = create_sppmi_mtx(network_spat, 1)

    n = W1.shape[0]
    B = np.eye(n, dim)
    Y1 = np.zeros([dim, n])
    Y2 = np.zeros([dim, n])
    Y3 = np.zeros([n, n])
    H = B.T.copy()
    E = np.zeros([dim, n])
    Z = np.zeros([n, n])
    J = np.zeros([n, n])

    mu = 0.1
    pho = 1.5
    max_mu = 1e4

    from network import softth, solve_l1l2

    epoch_iter = trange(100)
    for epoch in epoch_iter:
        F1 = np.linalg.solve(a=B.T @ B, b=B.T @ W1)
        F2 = np.linalg.solve(B.T @ B, B.T @ W2)
        F1 = np.maximum(F1, 0)
        F2 = np.maximum(F2, 0)

        # Update B
        B = ((W1 @ F1.T) + (W2 @ F2.T) + Y2.T + mu * H.T) @ np.linalg.inv((F1 @ F1.T + F2 @ F2.T + mu * np.eye(dim)))
        B = np.maximum(0, B)

        # Update H
        # H = H * (((E - Y1 / mu) @ (np.eye(n) - Z).T + B.T - Y2 / mu) / (H @ (np.eye(n) - Z) @ (np.eye(n) - Z).T + H))
        H = ((E - Y1 / mu) @ (np.eye(n) - Z).T + B.T - Y2 / mu) @ np.linalg.inv(
            (np.eye(n) - Z) @ (np.eye(n) - Z).T + np.eye(n))

        # Update Z
        # Z = np.linalg.inv(H.T @ H + np.eye(n)) @ (H.T @ (H - E) + (H.T @ Y1 - Y3) / mu + (J - np.diag(np.diag(J))))
        Z = np.linalg.solve(a=H.T @ H + np.eye(n), b=(H.T @ (H - E) + (H.T @ Y1 - Y3) / mu + (J - np.diag(np.diag(J)))))

        # Update E
        A = H - H @ Z + (Y1 / mu)
        E = solve_l1l2(A, lamb / mu)

        # Update J
        J = softth(Z + Y3 / mu, gamma / mu)

        # Update multipliers
        Y1 = Y1 + mu * (H - H @ Z - E)
        Y2 = Y2 + mu * (H - B.T)
        Y3 = Y3 + mu * (Z - J + np.diag(np.diag(J)))

        mu = min(pho * mu, max_mu)

        # Convergence conditions
        err1 = np.linalg.norm(H - H @ Z - E, np.inf)
        err2 = np.linalg.norm(H - B.T, np.inf)
        err3 = np.linalg.norm(Z - J + np.diag(np.diag(J)), np.inf)
        max_err = max(err1, err2, err3)

        total_loss = err1 + err2 + err3

        if max_err < 5e-2:
            break

        epoch_iter.set_description(
            f"# Epoch {epoch}, loss: {total_loss.item():.3f}")

    Z = 0.5 * (np.abs(Z) + np.abs(Z.T))
    return Z
