import sys
sys.path.append('/Users/zhanganran/Desktop/HACD')
import json
from train.modu_pre import Modutrain_hacd
from train.comm_pre import Commtrain_hacd
import argparse
from torch_geometric.utils import coalesce
from datetime import datetime
import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
from utils import preprocess_adj, get_info, load, sparse_to_tuple, compute_Q, build_hetero, get_meta_paths
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def write2file(comms, filename):
    with open(filename, 'w') as fh:
        content = '\n'.join([', '.join([str(i) for i in com]) for com in comms])
        fh.write(content)


def read4file(filename):
    with open(filename, "r") as file:
        pred = [[int(node) for node in x.split(', ')] for x in file.read().strip().split('\n')]
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--hidden", type=list, help="training hidden size", default=32)
    parser.add_argument("--output_dim", type=int, help="training hidden size", default=32)
    parser.add_argument("--dropout_h", type=list, help="dropout rate", default=0.2)
    parser.add_argument("--margin", type=float, help="margin loss", default=0.4)
    parser.add_argument("--fine_ratio", dest="fine_ratio", type=float, help="fine-grained sampling ratio", default=0.0)
    parser.add_argument("--comm_max_size", type=int, help="Community max size", default=12)
    parser.add_argument("--dataset", type=str, help="dataset", default="cora")
    parser.add_argument("--field", type=list, help='high_hop adj weight', default=[1, 1e-1])
    parser.add_argument("--learning_rate", type=float, help='learning rate', default=0.01)
    parser.add_argument("--weightdecay", type=int, help='weight decay', default=0)
    parser.add_argument("--num_epoch", type=int, help='training epoch', default=400)
    parser.add_argument("--par1", type=float, help='loss para', default=1e3)
    parser.add_argument("--par2", type=float, help='loss para', default=1e3)
    parser.add_argument("--print_yes", type=int, help='print_yes', default=1)
    parser.add_argument("--print_intv", type=int, help='print_intv', default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_heads", type=int, help="num of attention heads", default=2)
    parser.add_argument('--num_clusters', type=int, default=7)

    args = parser.parse_args()
    print("Using {} dataset".format(args.dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('= ' * 20)
    print('##  Starting Time:', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    args.pretrain_path = f'./pretrain/predaegc_{args.dataset}_{args.num_epoch}.pkl'

    adj, features, features_c, labels, valid_labels, valid_indices, num_unique_nodes = load(args.dataset)  

    features_c = sparse_to_tuple(features_c)
    i = torch.from_numpy(features_c[0]).long().to(device)
    v = torch.from_numpy(features_c[1]).to(device)
    features_c = torch.sparse.FloatTensor(i.t(), v, features_c[2]).to(device)
    features_c = features_c.to(torch.float32)

    info = get_info(adj, args.field)
    info = info.to(device)

    supports = preprocess_adj(adj)
    
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    i = torch.from_numpy(adj_label[0]).long().to(device)
    v = torch.from_numpy(adj_label[1]).to(device)
    adj_label = torch.sparse.FloatTensor(i.t(), v, adj_label[2])
    weight_mask = adj_label.to_dense() == 1
    weight_tensor = torch.ones_like(weight_mask).to(device)
    weight_tensor[weight_mask] = 10

    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

    num_features_c_nonzero = features_c._nnz()
    featc_dim = features_c.shape[1]
    
    adj_coo = adj.tocoo()
    edge_index = torch.from_numpy(np.array([adj_coo.row, adj_coo.col], dtype=np.int64))
    edge_index, _ = coalesce(edge_index, None, adj.shape[0], adj.shape[0])

    edge_index = edge_index.to(device)

    pyg_g = build_hetero(args.dataset)
    meta_paths = get_meta_paths(pyg_g)

    num_heads = [1]

    Comm_obj = Modutrain_hacd(meta_paths, featc_dim, args, num_heads, adj, features_c, pyg_g, info, weight_tensor)
    emb = Comm_obj.train()

    Comm_obj = Commtrain_hacd(meta_paths, featc_dim, args, num_heads, adj, features_c, valid_labels, valid_indices, pyg_g, info, weight_tensor)
    emb = Comm_obj.train()