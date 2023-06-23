import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def soft(x, T):
    if np.sum(np.abs(T)) == 0:
        y = x
    else:
        y = np.maximum(np.abs(x) - T, 0)
        y = np.sign(x) * y
    return y


def sparse_self_representation(x, alpha=0.6, beta=0.1):
    # x \in R^{d \times n}
    max_epoch = 100
    init_W = cosine_similarity(x.T)
    init_W = kneighbors_graph(init_W, n_neighbors=15, mode='connectivity', include_self=False).toarray()
    n = x.shape[1]
    T1 = np.zeros((n, n))

    C = init_W
    J1 = C
    mu = 50
    D = np.diag(np.sum(init_W, axis=1))

    for epoch in tqdm(range(max_epoch)):
        # 更新 C 矩阵
        C = C * ((x.T @ x + mu * (J1 - np.diag(np.diag(J1))) - T1 + beta * init_W @ C) /
                 (x.T @ x @ C + mu * C + beta * D @ C) + 1e-18)
        C = C - np.diag(np.diag(C))
        C[C < 0] = 0
        # 计算 J1 矩阵
        J1 = np.array(soft(C + T1 / mu, alpha / mu))
        J1 = J1 - np.diag(np.diag(J1))
        # 更新 T1 矩阵
        T1 = T1 + mu * (C - J1)

        # 计算误差
        err = np.linalg.norm(x - x @ C, 'fro')
        if err < 1e-2:
            break

        print('Epoch is: {}, loss={}'.format(epoch, err))
    return C