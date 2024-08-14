import sys

import sklearn
sys.path.append('/Users/zhanganran/Desktop/HACD-wsdm')
import argparse
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

from model.Cluster import CommC_hacd

from utils.metrics import eva
from utils.evalution import result
from utils.help_func import *


def target_distribution(q):
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

def dot_product_decode(Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class Commtrain_hacd:
    def __init__(self, meta_paths, input_dim, args, num_heads, adj, features, labels, valid_indices, g, info, weight_tensor):
        self.args = args
        self.input_dim = input_dim
        self.adj = adj
        self.features = features
        self.labels = labels
        self.valid_indices = valid_indices
        self.info = info
        self.weight_tensor = weight_tensor
        self.g = g
        self.model = CommC_hacd(meta_paths, input_dim, args, num_heads)
        self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)


    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.train()
        total_time_start = time.time()

        y = self.labels

        with torch.no_grad():
            _, embedding = self.model.pre_model(self.g.to(device), self.features.to(device))

        valid_embeddings = embedding[self.valid_indices]

        kmeans = KMeans(n_clusters=self.args.num_clusters, n_init=20)
        y_pred = kmeans.fit_predict(valid_embeddings.data.cpu().numpy())

        '''dbscan = DBSCAN(eps=0.5, min_samples=5)
        y_pred = dbscan.fit_predict(valid_embeddings)'''
        #y_pred = dbscan.fit_predict(embedding.data.cpu().numpy())

        #self.model.cluster_layer.data = torch.tensor(dbscan.cluster_centers_).to(device)
        #result(y, y_pred, 'pretrain')

        for epoch in range(1, self.args.num_epoch):
            time_start = time.time()
            self.opt.zero_grad()

            prob, embedding, q = self.model(self.g.to(device), self.features.to(device))
            p = target_distribution(q.detach())

            I = torch.eye(self.info.shape[0]).to(device)
            Q = modularity(self.info - I * self.info, prob)
            R = reconstruct(prob, self.info, self.weight_tensor)
            loss = self.args.par1 * Q + self.args.par2 * R
            
            loss.backward()
            self.opt.step()

            time_end = time.time()
            time_epoch = time_end - time_start
            if self.args.print_yes and epoch % self.args.print_intv == 0:
                prob, embedding, Q = self.model(self.g.to(device), self.features.to(device))

                Q = Q[self.valid_indices]
                
                q = Q.detach().data.cpu().numpy().argmax(1)  # Q
                acc_1, nmi_1, ari_1, f1_1 = eva(y, q, epoch)
                acc_2, nmi_2, ari_2, f1_2 = result(q, y, epoch)
                print(f"epoch {epoch}: time {time_epoch:.4f}, acc_1 {acc_1:.4f}, nmi_1 {nmi_1:.4f}, ari_1 {ari_1:.4f}, f1_1 {f1_1:.4f}")
                print(f"epoch {epoch}: time {time_epoch:.4f}, acc_2 {acc_2:.4f}, nmi_2 {nmi_2:.4f}, ari_2 {ari_2:.4f}, f1_2 {f1_2:.4f}")

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        self.model.eval()
        prob, embedding, Q = self.model(self.g, self.features)
        Q = Q[self.valid_indices]
        embedding = embedding.detach().to("cpu")
        q = Q.detach().data.cpu().numpy().argmax(1)  # Q
        acc_1, nmi_1, ari_1, f1_1 = eva(y, q, epoch)
        acc_2, nmi_2, ari_2, f1_2 = result(q, y, epoch)
        print(f"HACD: acc_1 {acc_1:.4f}, nmi_1 {nmi_1:.4f}, ari_1 {ari_1:.4f}, f1_1 {f1_1:.4f}")
        print(f"HACD: acc_2 {acc_2:.4f}, nmi_2 {nmi_2:.4f}, ari_2 {ari_2:.4f}, f1_2 {f1_2:.4f}")
        print("Total time: %.4f" % total_time)