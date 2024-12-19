import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx

from typing import List, Tuple, Callable, Dict, Union

import eyefeatures.features.scanpath_complex as eye_complex


def add_new_similarity_line(dist_matrix, object_ids, df_with_scanpath, new_scanpath, dist_metric: Callable, x: str, y: str, pk: str):
    new_dist_matrix = dist_matrix.copy()
    for i, inst_tr in enumerate(object_ids[:-1]):
        # train_scanpath = df_with_scanpath[
        #     (df_with_scanpath["participant_id"] == inst_tr[0]) & (df_with_scanpath["image_id"] == inst_tr[1])][
        #     ["x", "y"]]
        train_scanpath = df_with_scanpath[df_with_scanpath[pk] == inst_tr][[x, y]]
        dist = dist_metric(train_scanpath, new_scanpath)

        new_dist_matrix[-1, i] = new_dist_matrix[i, -1] = dist

    new_dist_matrix /= np.max(new_dist_matrix)

    n, _ = new_dist_matrix.shape
    sim_matrix = np.zeros(new_dist_matrix.shape)
    for i in range(n):
        for j in range(i + 1, n):
            sim_matrix[i, j] = sim_matrix[j, i] = 1 / new_dist_matrix[i, j]

    sim_matrix = sim_matrix / sim_matrix.max() + np.eye(n)
    sim_matrix = pd.DataFrame(index=object_ids, columns=object_ids, data=sim_matrix)

    return sim_matrix

def dist2similarity(distance: float) -> float:
    return 1 / distance


def get_distance_matrix(df: pd.DataFrame, dist_metric: Callable, x: str, y: str, pk: List[str]) -> Tuple[List[str], pd.DataFrame]:
    scanpaths_df = df[[*pk, x, y]]
    object_ids, scanpaths = [], []
    for object_id, scanpath_df in scanpaths_df.groupby(pk):
        object_ids.append(object_id)
        scanpaths.append(scanpath_df[[x, y]])

    dist_matrix = eye_complex.get_dist_matrix(scanpaths, dist_metric=dist_metric).values
    return dist_matrix, object_ids

def get_similarity_matrix(df: pd.DataFrame, dist_metric: Callable, x: str, y: str, pk: List[str]) -> Tuple[List[str], pd.DataFrame]:
    # scanpaths_df = df[["participant_id", "image_id", "x", "y"]]
    # scanpaths_df = df[[*pk, x, y]]
    # object_ids, scanpaths = [], []
    # for object_id, scanpath_df in scanpaths_df.groupby(pk):
    #     object_ids.append(object_id)
    #     scanpaths.append(scanpath_df[[x, y]])
    #
    # dist_matrix = eye_complex.get_dist_matrix(scanpaths, dist_metric=dist_metric).values
    dist_matrix, object_ids = get_distance_matrix(df, dist_metric=dist_metric, x=x, y=y, pk=pk)
    max_dist = dist_matrix.max()
    dist_matrix /= max_dist
    n, _ = dist_matrix.shape

    sim_matrix = np.zeros(dist_matrix.shape)
    for i in range(n):
        for j in range(i + 1, n):
            sim_matrix[i, j] = sim_matrix[j, i] = dist2similarity(dist_matrix[i, j])

    sim_matrix = sim_matrix / sim_matrix.max() + np.eye(n)
    sim_matrix = pd.DataFrame(index=object_ids, columns=object_ids, data=sim_matrix)
    return sim_matrix

def make_adjacency_from_similarity(similarity_matrix: pd.DataFrame, threshold: float, weighted: bool = True) -> pd.DataFrame:
    assert 0 <= threshold <= 1.0
    n, _ = similarity_matrix.shape
    adjacency_matrix = similarity_matrix.copy()
    adjacency_matrix -= np.eye(n)  # remove loops
    adjacency_matrix[adjacency_matrix < threshold] = 0
    if not weighted:
        adjacency_matrix[adjacency_matrix >= threshold] = 1
    return adjacency_matrix

def get_scaled_size(adjacency_matrix: pd.DataFrame, is_directed: bool = True) -> float:
    V, _ = adjacency_matrix.shape
    size = (adjacency_matrix.values != 0).sum()
    scaled_size = size / (V * (V - 1))
    if not is_directed:
        scaled_size *= 2
    return scaled_size

def plot_graph_size(similarity_matrices: Dict[str, pd.DataFrame], thresholds: List[float]):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (dist_name, similarity_matrix) in enumerate(similarity_matrices.items()):
        scaled_sizes = []
        for threshold in thresholds:
            adjacency_matrix = make_adjacency_from_similarity(similarity_matrix, threshold)
            scaled_sizes.append(get_scaled_size(adjacency_matrix))
        sns.lineplot(x=thresholds, y=scaled_sizes, label=dist_name, ax=ax)
        ax.set(xlabel="Thresholds", ylabel="# of Edges (scaled)")

def find_threshold_for_size(similarity_matrix: pd.DataFrame, required_size: float, atol: float = 1e-6, max_iters: int = 20) -> float:
    r = 1.0
    l = 0.0
    best_threshold = -1
    best_diff      = 1
    n_iters        = 0
    while True:
        n_iters += 1
        m = (r + l) / 2
        adjacency_matrix = make_adjacency_from_similarity(similarity_matrix, threshold=m)
        scaled_size = get_scaled_size(adjacency_matrix)
        if scaled_size > required_size:
            l = m
        else:
            r = m
        diff = abs(scaled_size - required_size)
        if diff < best_diff:
            best_diff = diff
            best_threshold = m
        if best_diff < atol or n_iters >= max_iters:
            break
    return best_threshold

def adjacency2edgelist(adjacency_matrix: pd.DataFrame) -> pd.DataFrame:
    n, _ = adjacency_matrix.shape
    headers  = ["source", "target", "weight"]
    edgelist = []
    for i in range(n):
        for j in range(i + 1, n):
            source = adjacency_matrix.index[i]
            target = adjacency_matrix.columns[j]
            weight = adjacency_matrix.loc[source, target]
            edgelist.append([source, target, weight])
    return pd.DataFrame(data=edgelist, columns=headers)

def make_graph_from_adjacency(adjacency_matrix: pd.DataFrame) -> nx.Graph:
    # edgelist = adjacency2edgelist(adjacency_matrix)
    graph    = nx.from_pandas_adjacency(adjacency_matrix, create_using=nx.Graph)
    return graph

def make_graph_from_similarity(similarity_matrix: pd.DataFrame, size: float = None, threshold: float = None) -> nx.Graph:
    assert size is not None or threshold is not None
    if threshold is None:
        threshold = find_threshold_for_size(similarity_matrix, required_size=size)
    adjacency_matrix = make_adjacency_from_similarity(similarity_matrix, threshold)
    return make_graph_from_adjacency(adjacency_matrix)

def calc_inter(similarity_matrices: Dict[str, pd.DataFrame], size: float) -> pd.DataFrame:
    headers = list(similarity_matrices.keys())
    n = len(similarity_matrices)
    V, _ = similarity_matrices[headers[0]].shape
    inter_matrix = np.zeros((n, n))
    for i, (dist_name1, sm1) in enumerate(similarity_matrices.items()):
        for j, (_, sm2) in enumerate(similarity_matrices.items()):
            graph_i = make_graph_from_similarity(sm1, size=size)   # undirected graph
            graph_j = make_graph_from_similarity(sm2, size=size)   # undirected graph
            inter   = nx.intersection(graph_i, graph_j)
            real_size_i = len(graph_i.edges) * 2 / (V * (V - 1))
            real_size_j = len(graph_j.edges) * 2 / (V * (V - 1))
            real_size_est = (real_size_i + real_size_j) / 2
            inter_matrix[i, j] = inter_matrix[j, i] = len(inter.edges) / real_size_est
            if i == j:
                assert len(inter.edges) == len(graph_i.edges) == len(graph_j.edges)
    inter_matrix *= 2 / (V * (V - 1))
    return pd.DataFrame(data=inter_matrix, index=headers, columns=headers)
