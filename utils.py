import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import logging
from graph import Graph
import math


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def generate_adjlist_with_all_edges(G):
    delimiter = ' '
    for s, nbrs in G.adjacency():
        line = str(s) + delimiter
        for t, data in nbrs.items():
            line += str(t) + delimiter
        yield line[: -len(delimiter)]
    return line


def load_data_from_npz(dataset_str):    # photo, cs, computer
    A = np.load('../Datasets/{}.npz'.format(dataset_str))

    adj = sp.csr_matrix((A.f.adj_data, A.f.adj_indices, A.f.adj_indptr), shape=A.f.adj_shape)
    adj = adj + adj.transpose()     # transfer to symmetric matrix
    adj[adj > 1] = 1

    # find the isolated nodes
    adjlist = {}  # adj list of the graph
    iso_nodes = []
    G = nx.from_numpy_array(adj.toarray())
    for line in generate_adjlist_with_all_edges(G):
        l = list(map(int, line.split()))
        adjlist[l[0]] = l[1:]
        if l[1:]==[]:
            iso_nodes.append(l[0])

    # feature of nodes, features of isolated nodes are set to 0-vector.
    x = sp.csr_matrix((A.f.attr_data, A.f.attr_indices, A.f.attr_indptr), shape=A.f.attr_shape)
    x = x.toarray()
    x[iso_nodes, :] = 0
    x = sp.csr_matrix(x)

    # one-hot labels of nodes, labels of isolated nodes are set to 0.
    labels = A.f.labels
    labels[iso_nodes] = 0
    n_inst = labels.shape[0]
    n_class = max(labels) + 1
    y = tf.one_hot(labels, n_class, axis=1)
    sess = tf.Session()
    y = y.eval(session=sess)

    # divide into: training sets (30 samples for each class), test sets
    train_idx = []
    val_idx = []
    test_idx = []
    for c in range(n_class):
        idx = np.argwhere(labels == c).ravel()
        idx = list(set(idx) - set(iso_nodes))
        np.random.shuffle(idx)
        train_idx.append(idx[:30])
        val_idx.append(idx[30:60])
        test_idx.append(idx[60:])

    train_idx = np.sort([b for a in train_idx for b in a])
    train_mask = sample_mask(train_idx, y.shape[0])

    val_idx = np.sort([b for a in val_idx for b in a])
    val_mask = sample_mask(val_idx, n_inst)

    test_idx = np.sort([b for a in test_idx for b in a])
    test_mask = sample_mask(test_idx, n_inst)

    y_train = np.zeros([n_inst, n_class])
    y_train[train_mask] = y[train_mask]

    y_val = np.zeros([n_inst, n_class])
    y_val[val_mask] = y[val_mask]

    y_test = np.zeros([n_inst, n_class])
    y_test[test_mask] = y[test_mask]

    # save adjlist to files
    f = open('../Datasets/{}/ind.{}.graph'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(adjlist, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.adj'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(adj, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.x'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(x, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.y_train'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(y_train, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.y_val'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(y_val, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.y_test'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(y_test, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.train_mask'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(train_mask, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.val_mask'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(val_mask, f)
    f.close()
    f = open('../Datasets/{}/ind.{}.test_mask'.format(dataset_str, dataset_str), 'wb')
    pkl.dump(test_mask, f)
    f.close()

    return adj, x, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data0(dataset_str, n_per_class):
    names = ['adj', 'x', 'y_train', 'y_val', 'y_test', 'train_mask', 'val_mask', 'test_mask']
    objects = []
    dir_str = str(n_per_class) + 'PerClass'
    for i in range(len(names)):
        with open("../Datasets/{}/ind.{}.{}".format(dir_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    adj, x, y_train, y_val, y_test, train_mask, val_mask, test_mask = tuple(objects)

    return adj, x, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_random(dataset_str, n_per_class):
    names = ['adj', 'x', 'y_train', 'y_val', 'y_test', 'train_mask', 'val_mask', 'test_mask']
    objects = []
    dir_str = 'data'
    for i in range(len(names)):
        with open("../Datasets/{}/ind.{}.{}".format(dir_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    adj, x, y_train, y_val, y_test, train_mask, val_mask, test_mask = tuple(objects)

    # randomly sample m (<20/30) nodes for each class
    if n_per_class is not None and n_per_class < 20:
        id_train = []
        for i in range(y_train.shape[1]):
            id_cur_class = np.argwhere(y_train[:, i] == 1).squeeze()
            np.random.shuffle(id_cur_class)
            id_train.append(id_cur_class[:n_per_class])
        id_train = np.sort(np.array(id_train).ravel())
        train_mask = sample_mask(id_train, y_train.shape[0])
        tmp = np.zeros(y_train.shape)
        tmp[train_mask, :] = y_train[train_mask, :]
        y_train = tmp

    return adj, x, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_rate(dataset, public, percent, seed_k):
    dataset_str = dataset
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../Datasets/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    test_idx_reorder = parse_index_file("../Datasets/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adjlist = {}  # adj list of the graph
    iso_nodes = []
    G = nx.from_numpy_array(adj.toarray())
    for line in generate_adjlist_with_all_edges(G):
        l = list(map(int, line.split()))
        adjlist[l[0]] = l[1:]
        if l[1:] == []:
            iso_nodes.append(l[0])

    idx_test_public = test_idx_range.tolist()
    idx_train, idx_val, idx_test = split_dataset(idx_test_public, len(labels), np.argmax(labels, 1), dataset, public,
                                                 percent, seed_k)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # return adj, features, labels, idx_train, idx_val, idx_test
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def split_dataset(idx_test_public, num_nodes, labels, dataset, public, percent, seed_k):
    '''test-index，nodes，labels，dataset，1：20 pre class；2：nodes；0：labels rates，GPU'''
    random.seed(seed_k)
    if public == 1:
        if dataset == 'cora':
            idx_train, idx_val, idx_test = range(140), range(140, 640), idx_test_public
        elif dataset == 'citeseer':
            idx_train, idx_val, idx_test = range(120), range(120, 620), idx_test_public
        elif dataset == 'pubmed':
            idx_train, idx_val, idx_test = range(60), range(60, 560), idx_test_public
    elif public == 0:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels)
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train, random.sample(list(np.where(labels == c)[0].astype(int)),
                                                            math.ceil(np.where(labels == c)[0].shape[0] * percent))])
        others = np.delete(all_data.astype(int), idx_train.astype(int))
        for c in all_class:
            idx_val = np.hstack([idx_val, random.sample(list(np.where(labels[others] == c)[0].astype(int)),
                                                        math.ceil(500 / all_class.shape[0]))])
        others = np.delete(others.astype(int), idx_val.astype(int))
        for c in all_class:
            idx_test = np.hstack([idx_test, random.sample(list(np.where(labels[others] == c)[0].astype(int)),
                                                          min(math.ceil(1000 / all_class.shape[0]),
                                                              np.where(labels[others] == c)[0].astype(int).shape[0]))])

        idx_train = idx_train.astype(int).tolist()
        idx_val = idx_val.astype(int).tolist()
        idx_test = idx_test.astype(int).tolist()

    return idx_train, idx_val, idx_test


def load_data(dataset_str, n_per_class=None):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../Datasets/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)    # x - train, tx - test, allx - out of test
    test_idx_reorder = parse_index_file("../Datasets/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer' or dataset_str=='photo':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    adjlist = {}  # adj list of the graph
    iso_nodes = []
    G = nx.from_numpy_array(adj.toarray())
    for line in generate_adjlist_with_all_edges(G):
        l = list(map(int, line.split()))
        adjlist[l[0]] = l[1:]
        if l[1:] == []:
            iso_nodes.append(l[0])

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]

    # randomly sample m (<20) nodes for each class
    if n_per_class is not None and n_per_class < 20:
        id_train = []
        for i in range(labels.shape[1]):
            id_cur_class = np.argwhere(y_train[:, i] == 1).squeeze()
            np.random.shuffle(id_cur_class)
            id_train.append(id_cur_class[:n_per_class])
        id_train = np.sort(np.array(id_train).ravel())
        train_mask = sample_mask(id_train, labels.shape[0])
        y_train = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]

    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # # save adjlist to files
    # dir_str = str(n_per_class) + 'PerClass'
    # # dir_str = 'data'
    # f = open('../Datasets/{}/ind.{}.graph'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(graph, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.adj'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(adj, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.x'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(features, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.y_train'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(y_train, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.y_val'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(y_val, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.y_test'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(y_test, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.train_mask'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(train_mask, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.val_mask'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(val_mask, f)
    # f.close()
    # f = open('../Datasets/{}/ind.{}.test_mask'.format(dir_str, dataset_str), 'wb')
    # pkl.dump(test_mask, f)
    # f.close()

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def CalCLass01Mat(y_train, train_mask):  #
    """Mark two samples in the training set whether are from same class, obtain all-sample-size matrix"""
    y = np.argmax(y_train, axis=1)
    train_idx = np.argwhere(train_mask == False)
    y[train_idx] = -1
    num_classes = np.max(y) + 1
    mat01 = np.zeros([np.shape(y_train)[0], np.shape(y_train)[0]])
    for i in range(num_classes):
        pos = np.argwhere(y == i)
        #        print(np.shape(mat01))
        for j in range(np.shape(pos)[0]):
            mat01[pos[j, 0], pos[:, 0]] = 1
    mat01[[i for i in range(np.shape(y_train)[0])], [i for i in range(np.shape(y_train)[0])]] = 0
    return mat01


def CalIntraClassMat01(y):
    """For m samples in training set, mark two samples whether have same or different labels, obtain 2 matrices"""
    num1 = np.shape(y)[0]
    num_classes = np.max(y) + 1
    mat01_intra = np.zeros([num1, num1])
    mat01_inter = np.ones([num1, num1])
    for class_idx in range(int(num_classes)):
        pos = np.argwhere(y == class_idx)
        for pos_idx in range(np.shape(pos)[0]):
            mat01_intra[pos[pos_idx, 0], pos[:, 0]] = 1
    mat01_inter -= mat01_intra  # different class
    mat01_intra -= np.eye(num1)  # inner class
    return [mat01_intra, mat01_inter]


def construct_feed_dict(features, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def read_graph_from_adj(dataset_name):
    '''Assume idx starts from *1* and are continuous. Edge shows up twice. Assume single connected component.'''
#    logging.info("Reading graph from metis...")
    dir_str = 'data'
    with open("../Datasets/{}/ind.{}.adj".format(dir_str, dataset_name), 'rb') as f:
        if sys.version_info > (3, 0):
            adj = pkl.load(f, encoding='latin1')
        else:
            adj = pkl.load(f)
    with open("../Datasets/{}/ind.{}.graph".format(dir_str, dataset_name), 'rb') as f:
        if sys.version_info > (3, 0):
            in_file = pkl.load(f, encoding='latin1')
        else:
            in_file = pkl.load(f)
    weighted = False 
    node_num = adj.shape[0]
    edge_num = np.count_nonzero(adj.toarray()) * 2
    graph = Graph(node_num, edge_num)
    edge_cnt = 0
    graph.adj_idx[0] = 0
    for idx in range(node_num):
        graph.node_wgt[idx] = 1
        eles = in_file[idx]
        j = 0 
        while j < len(eles):
            neigh = int(eles[j])  #
            if weighted:
                wgt = float(eles[j+1])
            else:
                wgt = 1.0
            graph.adj_list[edge_cnt] = neigh # self-loop included.
            graph.adj_wgt[edge_cnt] = wgt
            graph.degree[idx] += wgt
            edge_cnt += 1
            if weighted:
                j += 2
            else:
                j += 1
        graph.adj_idx[idx+1] = edge_cnt
    graph.A = graph_to_adj(graph, self_loop=False)
    return graph, None


def graph_to_adj(graph, self_loop=False):
    '''self_loop: manually add self loop or not'''
    node_num = graph.node_num
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(0, node_num):
        for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i+1]):
            i_arr.append(i)
            j_arr.append(graph.adj_list[neigh_idx])
            data_arr.append(graph.adj_wgt[neigh_idx])
    adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    return adj


def cmap2C(cmap): # fine_graph to coarse_graph, matrix format of cmap: C: n x m, n>m.
    node_num = len(cmap)
    i_arr = []
    j_arr = []
    data_arr = []
    for i in range(node_num):
        i_arr.append(i)
        j_arr.append(cmap[i])
        data_arr.append(1)
    return sp.csr_matrix((data_arr, (i_arr, j_arr)))      

