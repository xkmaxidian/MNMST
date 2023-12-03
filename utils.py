from typing import Tuple, Union

import anndata
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
from scipy.sparse import issparse
import psutil
from matplotlib.collections import LineCollection


# 该方法，仅当使用半径搜索邻域节点时调用
def remove_greater_than(graph: csr_matrix,
                        threshold: float,
                        copy: bool = False,
                        verbose: bool = True):
    """
    Remove values greater than a threshold from a CSR matrix
    """
    if copy:
        graph = graph.copy()

    greater_indices = np.where(graph.data > threshold)[0]

    if verbose:
        print(f"CSR data field:\n{graph.data}\n"
              f"compressed indices of values > threshold:\n{greater_indices}\n")

    # delete the entries in data and index fields
    # -------------------------------------------

    graph.data = np.delete(graph.data, greater_indices)
    graph.indices = np.delete(graph.indices, greater_indices)

    # update the index pointer
    # ------------------------

    hist, _ = np.histogram(greater_indices, bins=graph.indptr)
    cum_hist = np.cumsum(hist)
    graph.indptr[1:] -= cum_hist

    if verbose:
        print(f"\nCumulative histogram:\n{cum_hist}\n"
              f"\n___ New CSR ___\n"
              f"pointers:\n{graph.indptr}\n"
              f"indices:\n{graph.indices}\n"
              f"data:\n{graph.data}\n")

    return graph


# 对使用KNN求得的近邻矩阵，使用该函数进行归一化
def row_normalize(graph: csr_matrix,
                  copy: bool = False,
                  verbose: bool = True):
    """
    Normalize a compressed sparse row (CSR) matrix by row
    """
    if copy:
        graph = graph.copy()

    data = graph.data

    for start_ptr, end_ptr in zip(graph.indptr[:-1], graph.indptr[1:]):

        row_sum = data[start_ptr:end_ptr].sum()

        if row_sum != 0:
            data[start_ptr:end_ptr] /= row_sum

        if verbose:
            print(f"normalized sum from ptr {start_ptr} to {end_ptr} "
                  f"({end_ptr - start_ptr} entries)",
                  np.sum(graph.data[start_ptr:end_ptr]))

    return graph


def generate_spatial_distance_graph(
        locations: np.ndarray,
        nbr_object: NearestNeighbors = None,
        num_neighbours: int = None,
        radius: Union[float, int] = None,
) -> csr_matrix:
    if nbr_object is None:
        nbrs = NearestNeighbors(algorithm='ball_tree').fit(locations)
    else:
        nbrs = nbr_object

    if num_neighbours is None:
        return nbrs.radius_neighbors_graph(radius=radius, mode='distance')
    else:
        assert isinstance(num_neighbours, int), (
            f"number of neighbours {num_neighbours} is not an integer"
        )

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbours,
                                          mode="distance")
        if radius is not None:
            assert isinstance(radius, (float, int)), (
                f"Radius {radius} is not an integer or float"
            )

            graph_out = remove_greater_than(graph_out, radius,
                                            copy=False, verbose=False)

        return graph_out


# 根据输入的坐标信息，为固定的邻居节点数目，生成权重邻接图，并归一化
def generate_spatial_weights_fixed_nbrs(
        locations: np.ndarray,
        num_neighbours: int = 10,
        decay_type: str = 'reciprocal',
        nbr_object: NearestNeighbors = None,
        verbose: bool = True,
) -> Tuple[csr_matrix, csr_matrix]:
    distance_graph = generate_spatial_distance_graph(
        locations, nbr_object=nbr_object, num_neighbours=num_neighbours, radius=None,
    )
    graph_out = distance_graph.copy()
    graph_out.data = 1 / graph_out.data
    return row_normalize(graph_out, verbose=verbose), distance_graph


def plot_edge_histogram(graph: csr_matrix,
                        ax: mpl.axes.Axes,
                        title: str = "edge weights",
                        bins: int = 100):
    """
    plot a histogram of the edge-weights a graph
    """
    counts, bins, patches = ax.hist(graph.data, bins=bins)

    median_dist = np.median(graph.data)
    mode_dist = bins[np.argmax(counts)]
    ax.axvline(median_dist, color="r", alpha=0.8)
    ax.axvline(mode_dist, color="g", alpha=0.8)
    ax.set_title("Histogram of " + title)

    print(f"\nEdge weights ({title}): "
          f"median = {median_dist}, mode = {mode_dist}\n")

    return median_dist, mode_dist


def matrix_to_adata(matrix, adata: anndata.AnnData) -> anndata.AnnData:
    """
    convert a matrix to adata object, by
     - duplicating the original var (per-gene) annotations and adding "_nbr"
     - keeping the obs (per-cell) annotations the same as original anndata that banksy matrix was computed from
    """
    var_nbrs = adata.var.copy()
    var_nbrs.index += "_nbr"
    nbr_bool = np.zeros((var_nbrs.shape[0] * 2,), dtype=bool)
    nbr_bool[var_nbrs.shape[0]:] = True
    print("num_nbrs:", sum(nbr_bool))

    var_combined = pd.concat([adata.var, var_nbrs])
    var_combined["is_nbr"] = nbr_bool
    # 尽可能多的保留原始adata的信息
    return anndata.AnnData(matrix, obs=adata.obs, var=var_combined, uns=adata.uns, obsm=adata.obsm)


def weighted_concatenate(cell_genes: Union[np.ndarray, csr_matrix],
                         neighbours: Union[np.ndarray, csr_matrix],
                         neighbourhood_contribution: float,
                         ) -> Union[np.ndarray, csr_matrix]:
    """
    Concatenate self- with neighbour- feature matrices
    with a given contribution towards disimilarity from the neighbour features (lambda).
    Assumes that both matrices have already been z-scored.
    Will do sparse concatenation if BOTH matrices are sparse.
    """
    cell_genes *= np.sqrt(1 - neighbourhood_contribution)
    neighbours *= np.sqrt(neighbourhood_contribution)

    if issparse(cell_genes) and issparse(neighbours):

        return sparse.hstack((cell_genes, neighbours))

    else:  # at least one is a dense array
        if issparse(cell_genes):
            cell_genes = cell_genes.todense()
        elif issparse(neighbours):
            neighbours = neighbours.todense()

        return np.concatenate((cell_genes, neighbours), axis=1)


def zscore(matrix: Union[np.ndarray, csr_matrix],
           axis: int = 0,
           ) -> np.ndarray:
    """
    Z-score data matrix along desired dimension
    """
    # 求矩阵中每一行的平均值
    E_x = matrix.mean(axis=axis)

    if issparse(matrix):
        # 矩阵中所有数据平方，再求平均值
        squared = matrix.copy()
        squared.data **= 2
        E_x2 = squared.mean(axis=axis)
        del squared

    else:

        E_x2 = np.square(matrix).mean(axis=axis)

    variance = E_x2 - np.square(E_x)
    zscored_matrix = (matrix - E_x) / np.sqrt(variance)
    if isinstance(zscored_matrix, np.matrix):
        zscored_matrix = np.array(zscored_matrix)
    # Ensure that there are no NaNs
    # (which occur when all values are 0, hence variance is 0)
    zscored_matrix = np.nan_to_num(zscored_matrix)
    return zscored_matrix


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :][dis_df.loc[index, :] > 0]
        nbs = dis_tmp[0: num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if self_pred in nbs_pred:
            if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
                refined_pred.append(v_c.idxmax())
            else:
                refined_pred.append(self_pred)
        else:
            refined_pred.append(v_c.idxmax())
    return refined_pred


def record_memory_usage():
    # 获取当前进程的内存信息
    process = psutil.Process()
    memory_info = process.memory_info()
    # 将字节转换为 GB
    memory_usage_gb = memory_info.rss / (1024 ** 3)
    return memory_usage_gb


def plot_graph_weights(locations,
                       graph,
                       theta_graph=None,  # azimuthal angles
                       max_weight=1,
                       markersize=1,
                       figsize=(8, 8),
                       title: str = None,
                       flip_yaxis: bool = False,
                       ax=None,
                       ) -> None:
    """
    Visualize weights in a spatial graph,
    heavier weights represented by thicker lines
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    edges, weights, theta = [], [], []

    if theta_graph is not None:
        assert isinstance(theta_graph, csr_matrix)
        assert theta_graph.shape[0] == graph.shape[0]

    for start_node_idx in range(graph.shape[0]):

        ptr_start = graph.indptr[start_node_idx]
        ptr_end = graph.indptr[start_node_idx + 1]

        for ptr in range(ptr_start, ptr_end):
            end_node_idx = graph.indices[ptr]

            # append xs and ys as columns of a numpy array
            edges.append(locations[[start_node_idx, end_node_idx], :])
            weights.append(graph.data[ptr])
            if theta_graph is not None:
                theta.append(theta_graph.data[ptr])

    print(f"Maximum weight: {np.amax(np.array(weights))}\n")
    weights /= np.amax(np.array(weights))

    if theta_graph is not None:
        norm = mpl.colors.Normalize(vmin=min(theta), vmax=max(theta), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap="bwr")
        c = [mapper.to_rgba(t) for t in theta]
    else:
        c = "C0"

    line_segments = LineCollection(
        edges, linewidths=weights * max_weight, linestyle='solid', colors=c, alpha=0.7)
    ax.add_collection(line_segments)

    ax.scatter(locations[:, 0], locations[:, 1], s=markersize, c="gray", alpha=.6, )

    if flip_yaxis:  # 0 on top, typical of image data
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_aspect('equal', 'datalim')

    if title is not None:
        ax.set_title(title)