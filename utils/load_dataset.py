import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
import sys
import pickle as pkl
#from help_func import normalize

def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_true_labels(dataset_str, total_nodes):
    community_dict = {}
    node_to_label = {}
    
    with open(f"./dataset/{dataset_str}/{dataset_str}-cmty.txt") as file:
        for label, line in enumerate(file):
            nodes = line.strip().split()
            for node in nodes:
                node_to_label[int(node)] = label  

    true_labels = [-1] * total_nodes
    for node, label in node_to_label.items():
        true_labels[node] = label
    
    labels = np.array(true_labels)

    valid_indices = np.where(labels != -1)[0]

    valid_labels = labels[valid_indices]

    num_valid_nodes = len(valid_indices)

    unique_labels = np.unique(valid_labels)
    num_unique_nodes = len(unique_labels)

    return labels, valid_labels, valid_indices, num_unique_nodes

def load(dataset_str,normalize=True):
    root = '/Users/zhanganran/Desktop/HACD/dataset/'
    if dataset_str =='cora' or dataset_str =='citeseer' or dataset_str =='pubmed': 
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(root + "{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(root + "{}/ind.{}.test.index".format(dataset_str, dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = labels.argmax(axis=1)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        labels_sparse = sp.csr_matrix(np.eye(labels.max() + 1)[labels])
        features_c = sp.hstack([features, labels_sparse]).tocsr()

        features = features.tocsr()

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        valid_indices = np.where(labels != -1)[0]

        # 获取不为 -1 的标签
        valid_labels = labels[valid_indices]

        # 统计标签不为 -1 的节点数量
        num_valid_nodes = len(valid_indices)
        unique_labels = np.unique(valid_labels)
        num_unique_nodes = len(unique_labels)

        idx_test = test_idx_range.tolist()
        idx_test = np.array(idx_test)
        idx_train = np.array(range(len(y)))
        idx_val = np.array(range(len(y), len(y)+500))

    else:
        communities = open(f"./dataset/{dataset_str}/{dataset_str}-cmty.txt")
        edges = open(f"./dataset/{dataset_str}/{dataset_str}-ungraph.txt")

        communities = [[int(i) for i in x.split()] for x in communities]
        edges = [[int(i) for i in e.split()] for e in edges]
        edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

        nodes = {node for e in edges for node in e}
        mapping = {u: i for i, u in enumerate(sorted(nodes))}

        edges = [[mapping[u], mapping[v]] for u, v in edges]
        communities = [[mapping[node] for node in com] for com in communities]

        g = nx.Graph(edges)
        g.add_nodes_from(nodes)

        num_node = len(nodes)

        node_degree = [g.degree[node] for node in range(num_node)]

        features_c = np.zeros([num_node, 5], dtype=np.float32)
        features_c[:, 0] = np.array(node_degree).squeeze()
        features = []
        labels, valid_labels, valid_indices, num_unique_nodes = load_true_labels(dataset_str, num_node)
        idx_train = []
        idx_test = []
        idx_val = []

        new_graph = nx.Graph()
        for node in tqdm(range(num_node), desc="Feature Computation"):
            if len(list(g.neighbors(node))) > 0:
                neighbor_deg = features_c[list(g.neighbors(node)), 0]
                features_c[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

        if normalize:
            features_c = (features_c - features_c.mean(0, keepdims=True)) / (features_c.std(0, keepdims=True) + 1e-9)

        for node in tqdm(range(num_node), desc="Feature Augmentation"):
            node_feat = features_c[node, :].astype(np.float32)
            new_graph.add_node(node, node_feature=torch.from_numpy(node_feat))
        
        features_c = sp.csr_matrix(features_c)

        new_graph.add_edges_from(edges)
        adj = nx.adjacency_matrix(new_graph)

        '''num_samples = int(num_node * 0.4)
        sampled_nodes = np.random.choice(num_node, num_samples, replace=False)
        adj = adj[sampled_nodes][:, sampled_nodes]
        features_c = features_c[sampled_nodes]
        labels = labels[sampled_nodes]
        valid_indices = np.where(labels != -1)[0]
        valid_labels = labels[valid_indices]
        num_unique_nodes = len(np.unique(valid_labels))'''

        
    return adj, features, features_c, labels, valid_labels, valid_indices, num_unique_nodes

if __name__ == "__main__":
    adj, features, features_c, labels, valid_labels, valid_indices, num_unique_nodes = load('amazon')
    print(features_c)
    print(features_c.shape)
    '''print(adj)
    print(adj.shape)
    print(labels)
    print(labels.shape)
    print(num_unique_nodes)'''