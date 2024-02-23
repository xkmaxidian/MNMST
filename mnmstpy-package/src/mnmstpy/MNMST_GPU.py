import torch
import torch.nn.functional as F
from tqdm import trange
import torch.sparse as sp

from . network import create_sppmi_mtx


def soft_torch(x, T):
    if torch.sum(torch.abs(T)) == torch.tensor(0.):
        y = x
    else:
        y = torch.maximum(torch.abs(x) - T, torch.tensor(0.))
        y = torch.sign(x) * y
    return y


def create_sppmi_mtx_torch(G, k):
    # Calculating Degrees for each node
    node_degrees = torch.sum(G, dim=0)
    node_degrees2 = torch.sum(G, dim=1)
    W = torch.sum(node_degrees)

    sppmi = G.clone()

    indices = torch.nonzero(G)
    row = indices[:, 0]
    col = indices[:, 1]
    weights = G[row, col]

    for i in range(len(col)):
        score = torch.log(weights[i] * W / (node_degrees2[col[i]] * node_degrees[row[i]])) - torch.log(k)
        sppmi[col[i], row[i]] = max(score, torch.tensor(0.0))

    return sppmi


def solve_l1_l2_torch(W, lamb):
    nw = torch.norm(W, 2, dim=0, keepdim=True)
    e = (nw > lamb) * (nw - lamb) / (nw + 1e-18)
    return e * W


def softth_torch(F, lambda_val):
    U, S, Vt = torch.linalg.svd(F, full_matrices=False)
    Vt = Vt.t()

    svp = torch.sum(S > lambda_val).item()

    diagS = torch.clamp(S - lambda_val, min=0)

    if svp < 1:
        svp = 1

    E = U[:, :svp] @ torch.diag(diagS[:svp]) @ (Vt[:, :svp].t())
    return E


def sparse_self_representation_torch(x, init_w, alpha=1., beta=1., device='cpu'):
    # x \in R^{d \times n}
    max_epoch = 100
    n = x.shape[1]
    T1 = torch.zeros((n, n)).to(device)

    C = init_w.clone()
    J1 = C.clone()
    mu = 50.
    D = torch.diag(torch.sum(init_w, dim=0)).to(device)

    epoch_iter = trange(max_epoch)
    for epoch in epoch_iter:
        # 更新 C 矩阵
        C = C * ((x.t() @ x + mu * (J1 - torch.diag(torch.diag(J1))) - T1 + beta * init_w @ C) /
                 (x.t() @ x @ C + mu * C + beta * D @ C))
        C[C.isnan()] = 0
        C = C - torch.diag(torch.diag(C))
        # 计算 J1 矩阵
        J1 = soft_torch(C + T1 / mu, torch.tensor(alpha / mu))
        J1 = (J1 - torch.diag(torch.diag(J1)))
        # 更新 T1 矩阵
        T1 = T1 + mu * (C - J1)

        # 计算误差
        err = torch.norm(x - x @ C, 'fro')
        if err < 1e-2:
            break

        epoch_iter.set_description(f"# Epoch {epoch}, loss: {err.item():.3f}")
    C = 0.5 * (C.abs() + C.abs().t())
    return C


def MNMST_representation_gpu(network_exp, network_spat, lamb=10., gamma=10., dim=150, threshold=0.05, device='cpu'):
    # Variables init
    W1 = torch.from_numpy(create_sppmi_mtx(network_exp.detach().cpu().numpy(), 1)).float().to(device)
    W2 = torch.from_numpy(create_sppmi_mtx(network_spat.detach().cpu().numpy(), 1)).float().to(device)

    n = W1.shape[0]
    B = torch.eye(n, dim).to(device)
    Y1 = torch.zeros([dim, n]).to(device)
    Y2 = torch.zeros([dim, n]).to(device)
    Y3 = torch.zeros([n, n]).to(device)
    H = B.t().clone()
    E = torch.zeros([dim, n]).to(device)
    Z = torch.zeros([n, n]).to(device)
    J = torch.zeros([n, n]).to(device)
    I_d = torch.eye(dim).to(device)
    I_n = torch.eye(n).to(device)

    mu = 0.1
    pho = 1.5
    max_mu = 1e4

    epoch_iter = trange(100)
    for epoch in epoch_iter:
        F1 = torch.linalg.solve(B.t() @ B, B.t() @ W1)
        F2 = torch.linalg.solve(B.t() @ B, B.t() @ W2)
        F1 = F.relu(F1)
        F2 = F.relu(F2)

        # Update B
        B = ((W1 @ F1.t()) + (W2 @ F2.t()) + Y2.t() + mu * H.t()) @ torch.linalg.inv(
            (F1 @ F1.t() + F2 @ F2.t() + mu * I_d))
        B = F.relu(B)

        # Update H
        H = ((E - Y1 / mu) @ (I_n - Z).T + B.t() - Y2 / mu) @ torch.linalg.inv(
            (I_n - Z) @ (I_n - Z).t() + I_n)

        # Update Z
        Z = torch.linalg.solve(H.t() @ H + I_n,
                               (H.t() @ (H - E) + (H.t() @ Y1 - Y3) / mu + (J - torch.diag(torch.diag(J)))))

        # Update E
        A = H - H @ Z + (Y1 / mu)
        E = solve_l1_l2_torch(A, torch.tensor(lamb / mu))

        # Update J
        J = soft_torch(Z + Y3 / mu, torch.tensor(gamma / mu))

        # Update multipliers
        Y1 = Y1 + mu * (H - H @ Z - E)
        Y2 = Y2 + mu * (H - B.t())
        Y3 = Y3 + mu * (Z - J + torch.diag(torch.diag(J)))

        mu = min(pho * mu, max_mu)

        # Convergence conditions
        err1 = torch.norm(H - H @ Z - E, p=float('inf'))
        err2 = torch.norm(H - B.t(), p=float('inf'))
        err3 = torch.norm(Z - J + torch.diag(torch.diag(J)), p=float('inf'))
        max_err = max(err1, err2, err3)

        total_loss = err1 + err2 + err3

        if max_err < threshold:
            break

        epoch_iter.set_description(
            f"# Epoch {epoch}, loss: {total_loss.item():.3f}")

    Z = 0.5 * (torch.abs(Z) + torch.abs(Z.t()))
    return Z


def MNMST_representation_with_histology_gpu(network_his, network_exp, network_spat, lamb=10., gamma=10., dim=150, device='cpu'):
    W0 = torch.from_numpy(create_sppmi_mtx(network_his.detach().cpu().numpy(), 1)).float().to(device)
    W1 = torch.from_numpy(create_sppmi_mtx(network_exp.detach().cpu().numpy(), 1)).float().to(device)
    W2 = torch.from_numpy(create_sppmi_mtx(network_spat.detach().cpu().numpy(), 1)).float().to(device)

    n = W1.shape[0]
    B = torch.eye(n, dim).to(device)
    Y1 = torch.zeros([dim, n]).to(device)
    Y2 = torch.zeros([dim, n]).to(device)
    Y3 = torch.zeros([n, n]).to(device)
    H = B.t().clone()
    E = torch.zeros([dim, n]).to(device)
    Z = torch.zeros([n, n]).to(device)
    J = torch.zeros([n, n]).to(device)
    I_d = torch.eye(dim).to(device)
    I_n = torch.eye(n).to(device)

    mu = 0.1
    pho = 1.5
    max_mu = 1e4

    epoch_iter = trange(100)
    for epoch in epoch_iter:
        F0 = torch.linalg.solve(B.t() @ B, B.t() @ W0)
        F1 = torch.linalg.solve(B.t() @ B, B.t() @ W1)
        F2 = torch.linalg.solve(B.t() @ B, B.t() @ W2)
        F0 = F.relu(F0)
        F1 = F.relu(F1)
        F2 = F.relu(F2)

        # Update B
        B = ((W0 @ F0.t()) + (W1 @ F1.t()) + (W2 @ F2.t()) + Y2.t() + mu * H.t()) @ torch.linalg.inv(
            (F0 @ F0.t() + F1 @ F1.t() + F2 @ F2.t() + mu * I_d))
        B = F.relu(B)

        # Update H
        H = ((E - Y1 / mu) @ (I_n - Z).T + B.t() - Y2 / mu) @ torch.linalg.inv(
            (I_n - Z) @ (I_n - Z).t() + I_n)

        # Update Z
        Z = torch.linalg.solve(H.t() @ H + I_n,
                               (H.t() @ (H - E) + (H.t() @ Y1 - Y3) / mu + (J - torch.diag(torch.diag(J)))))

        # Update E
        A = H - H @ Z + (Y1 / mu)
        E = solve_l1_l2_torch(A, torch.tensor(lamb / mu))

        # Update J
        J = soft_torch(Z + Y3 / mu, torch.tensor(gamma / mu))

        # Update multipliers
        Y1 = Y1 + mu * (H - H @ Z - E)
        Y2 = Y2 + mu * (H - B.t())
        Y3 = Y3 + mu * (Z - J + torch.diag(torch.diag(J)))

        mu = min(pho * mu, max_mu)

        # Convergence conditions
        err1 = torch.norm(H - H @ Z - E, p=float('inf'))
        err2 = torch.norm(H - B.t(), p=float('inf'))
        err3 = torch.norm(Z - J + torch.diag(torch.diag(J)), p=float('inf'))
        max_err = max(err1, err2, err3)

        total_loss = err1 + err2 + err3

        if max_err < 5e-2:
            break

        epoch_iter.set_description(
            f"# Epoch {epoch}, loss: {total_loss.item():.3f}")

    Z = 0.5 * (torch.abs(Z) + torch.abs(Z.t()))
    return Z