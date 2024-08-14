import random
import numpy as np
import scipy.stats as stats
#from scipy.sparse import diags
import torch
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch
import scipy.sparse as sp
import networkx as nx
import torch.nn as nn
from sklearn.preprocessing import normalize


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^-0.5

'''def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten
    r_inv = np.array(r_inv)
    if np.issubdtype(r_inv.dtype, np.floating):
        r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx'''
    
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).bool()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1. / (1-rate))
    return out

def get_info(adj, field):
    current = adj
    stru = field[0] * adj
    for i in range(1, len(field)):
        current = current.dot(adj)
        stru += field[i] * current
    rowsum = np.array(stru.sum(1), dtype=np.float32)  # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1).flatten()  # 1/rowsum, [2708]
    r_inv[np.isinf(r_inv)] = 0.  # zero inf data
    r_mat_inv = sp.diags(r_inv)  # sparse diagonal matrix, [2708, 2708]
    stru = r_mat_inv.dot(stru)
    info = torch.Tensor(stru.todense())
    return info

def modularity(stru, prob):
    m = torch.sum(stru) / 2
    B = stru - (torch.sum(stru, dim=1, keepdim=True) *
                torch.transpose(torch.sum(stru, dim=1, keepdim=True), dim0=0, dim1=1)) / (2 * m)
    Q = torch.trace(torch.mm(torch.mm(torch.transpose(prob, dim0=0, dim1=1), B), prob)) / (2 * m)
    return -1 * Q

def reconstruct(prob, stru, weight_tensor):
    b_xent = nn.BCEWithLogitsLoss(weight=weight_tensor)
    R = b_xent(torch.matmul(prob, prob.t()), stru)
    return R

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def compute_Q(adj, prob):
    comm_labels = prob.argmax(dim=1).numpy()
    comm_dict = {}
    comm_index = 0
    for i in range(len(comm_labels)):
        if comm_labels[i] not in comm_dict:
            comm_dict[comm_labels[i]] = comm_index
            comm_index += 1
    comm_onehot = torch.zeros([len(comm_labels), len(np.unique(comm_labels))])
    for i in range(len(comm_labels)):
        comm_onehot[i][comm_dict[comm_labels[i]]] = 1
    Q = modularity(adj, comm_onehot)
    return -1 * Q

def get_M(adj):
    adj_numpy = adj.toarray()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)

def get_Q(z, cluster_layer, v):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / v)
        q = q.pow((v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q