import sys
sys.path.append('/Users/zhanganran/Desktop/HACD-wsdm')
import numpy as np
import torch
from torch_geometric.data import HeteroData
from utils.load_dataset import load
import scipy

def build_hetero(dataset):
    adj, features, features_c, labels, valid_labels, valid_indices, num_valid_nodes = load(dataset)

    if not isinstance(adj, scipy.sparse.coo_matrix):
        adj = adj.tocoo()

    paper_nodes = np.arange(len(labels))
    subject_nodes = np.arange(len(labels), len(labels) + len(np.unique(labels)))

    node_mapping = dict(zip(np.arange(len(labels)), paper_nodes))
    node_mapping.update(dict(zip(np.unique(labels), subject_nodes)))


    hetero_graph = HeteroData()

    hetero_graph['paper'].node_id = torch.arange(len(labels))
    hetero_graph['subject'].node_id = torch.arange(len(np.unique(labels)))

    hetero_graph['paper'].num_nodes = len(labels)
    hetero_graph['subject'].num_nodes = len(np.unique(labels))
    hetero_graph.metagraph = [
        ('paper', 'cites', 'paper'),
        ('paper', 'has_subject', 'subject')
    ]

    hetero_graph['cites'] = torch.tensor(np.array([adj.row, adj.col]), dtype=torch.long)
    hetero_graph['has_subject'] = torch.tensor(np.array([paper_nodes, labels]), dtype=torch.long)

    hetero_graph.edge_index_dict = {('paper', 'cites', 'paper'): hetero_graph['cites'], ('paper', 'has_subject', 'subject'): hetero_graph['has_subject']}

    return hetero_graph

def get_meta_paths(hetero_graph):
    meta_paths = []

    for edge in hetero_graph.metagraph:
        source_type, edge_type, target_type = edge
        meta_paths.append([source_type, edge_type, target_type])

    return meta_paths

if __name__ == "__main__":
    cora_hetero_graph = build_hetero('cora')
    meta_paths = get_meta_paths(cora_hetero_graph)
    print("Meta-paths:", meta_paths)
    print(cora_hetero_graph.edge_index_dict)
    print(cora_hetero_graph)
