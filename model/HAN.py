import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv

def validate_edge_index(edge_index, num_nodes):
    min_index = edge_index.min()
    max_index = edge_index.max()

    if min_index < 0 or max_index >= num_nodes:
        raise ValueError(f"Edge index out of valid node index range [0, {num_nodes-1}]. Found min index {min_index}, max index {max_index}")

    return edge_index

def clean_edge_index(edge_index, num_nodes):
    mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
    return edge_index[:, mask]


class SA_Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SA_Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )


    def forward(self, z, h):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        
        return (beta * z).sum(1)  # (N, D * K)


class HACDLayer(nn.Module):
    def __init__(self, meta_paths, input_dim, args, layer_num_heads, device='cuda:1'):
        super(HACDLayer, self).__init__()
        self.hidden_dim = args.hidden
        self.output_dim = args.output_dim
        self.dropout = args.dropout_h
        self.device = device

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    input_dim,
                    self.output_dim,
                    layer_num_heads,
                    self.dropout,
                    self.dropout,
                )
            )
        self.sa_attention = SA_Attention(
            in_size=self.output_dim * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

    def forward(self, g, h):
        sa_embeddings = []
        
        for i, meta_path in enumerate(self.meta_paths):
            edge_index = g.edge_index_dict[meta_path]
            num_nodes = h.shape[0]
            edge_index = clean_edge_index(edge_index, num_nodes)
            edge_index = validate_edge_index(edge_index, num_nodes)
            sa_embeddings.append(self.gat_layers[i](h, edge_index).flatten(1))
        sa_embeddings = torch.stack(
            sa_embeddings, dim=1
        )  # (N, M, D * K)

        return self.sa_attention(sa_embeddings, h)  # (N, D * K)


class HACDEncoder(nn.Module):
    def __init__(self, meta_paths, input_dim, args, num_heads):
        super(HACDEncoder, self).__init__()

        self.hidden_dim = args.hidden
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.dropout = args.dropout_h
        self.device = args.device
        self.sft = nn.Softmax(dim=1)

        self.layers = nn.ModuleList()
        self.layers.append(
            HACDLayer(meta_paths, input_dim, args, num_heads[0])
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HACDLayer(
                    meta_paths,
                    self.hidden_dim * num_heads[l - 1],
                    self.hidden_dim,
                    num_heads[l],
                    self.dropout,
                )
            )

        self.to(self.device)

    
    def forward(self, g, features):
        for gnn in self.layers:
            h = gnn(g, features)
            prob = self.sft(h)
        
        return prob, h
    
    def embedding(self, g, features):
        for gnn in self.layers:
            h = gnn(g, features)
        
        return h