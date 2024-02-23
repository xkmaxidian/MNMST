import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import trange
import scipy.sparse as sp


def soft_numpy(x, T):
    if np.sum(np.abs(T)) == 0.:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0.)
        y = np.sign(x) * y
    return y


def create_sppmi_mtx(G, k):
    node_degrees = np.array(G.sum(axis=0)).flatten()
    node_degrees2 = np.array(G.sum(axis=1)).flatten()
    W = np.sum(node_degrees)

    sppmi = G.copy()

    # Use a loop to calculate Wij*W/(di*dj)
    col, row, weights = sp.find(G)
    for i in range(len(col)):
        score = np.log(weights[i] * W / (node_degrees2[col[i]] * node_degrees[row[i]])) - np.log(k)
        sppmi[col[i], row[i]] = max(score, 0)

    return sppmi


def softth(F, lambda_val):
    temp = F.copy()
    U, S, Vt = np.linalg.svd(temp, full_matrices=False)
    Vt = Vt.T

    svp = len(np.flatnonzero(S > lambda_val))
    # svp = np.count_nonzero(S > lambda_val)

    diagS = np.maximum(0, S - lambda_val)

    if svp < 1:
        svp = 1

    E = U[:, :svp] @ np.diag(diagS[:svp]) @ Vt[:, 0: svp].T
    return E


def sparse_self_representation(x, init_w, alpha=1, beta=1):
    # x \in R^{d \times n}
    max_epoch = 100
    n = x.shape[1]
    T1 = np.zeros((n, n))

    C = init_w.copy()
    J1 = C.copy()
    mu = 50
    D = np.diag(np.sum(init_w, axis=0))

    epoch_iter = trange(max_epoch)
    for epoch in epoch_iter:
        # 更新 C 矩阵
        C = C * ((x.T @ x + mu * (J1 - np.diag(np.diag(J1))) - T1 + beta * init_w @ C) /
                 (x.T @ x @ C + mu * C + beta * D @ C))
        C[np.isnan(C)] = 0
        C = C - np.diag(np.diag(C))
        # 计算 J1 矩阵
        J1 = np.array(soft_numpy(C + T1 / mu, alpha / mu))
        J1 = J1 - np.diag(np.diag(J1))
        # 更新 T1 矩阵
        T1 = T1 + mu * (C - J1)

        # 计算误差
        err = np.linalg.norm(x - x @ C, 'fro')
        if err < 1e-2:
            break

        epoch_iter.set_description(f"# Epoch {epoch}, loss: {err.item():.3f}")
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    return C


def solve_l1l2(W, lambda_val):
    n = W.shape[0]
    E = W.copy()

    for i in range(n):
        E[i, :] = solve_l2(W[i, :], lambda_val)
    return E


def solve_l2(w, lambda_val):
    nw = np.linalg.norm(w)

    if nw > lambda_val:
        x = (nw - lambda_val) * w / nw
    else:
        x = np.zeros_like(w)
    return x
