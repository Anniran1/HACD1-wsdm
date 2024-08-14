import sys
sys.path.append('/Users/zhanganran/Desktop/HACD-wsdm')
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from model.HAN import HACDEncoder


class CommC_hacd(nn.Module):
    def __init__(self, meta_paths, input_dim, args, num_heads, v=1):
        super(CommC_hacd, self).__init__()
        self.args = args
        self.num_clusters = args.num_clusters
        self.output_dim = args.output_dim
        self.v = v

        self.pre_model = HACDEncoder(meta_paths, input_dim, args, num_heads)

        self.cluster_layer = Parameter(torch.Tensor(self.num_clusters, self.output_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, g, h):
        prob, embedding = self.pre_model(g, h)
        q = self.get_Q(embedding)

        return prob, embedding, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q