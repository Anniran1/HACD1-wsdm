import sys
sys.path.append('/Users/zhanganran/Desktop/HACD-wsdm')
import time
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
import dataset
from model.HAN import HACDEncoder
from utils.help_func import *
from utils.metrics import *


class Modutrain_hacd:
    def __init__(self, meta_paths, input_dim, args, num_heads, adj, features, g, info, weight_tensor):
        self.args = args
        self.input_dim = input_dim
        self.adj = adj
        self.features = features
        self.g = g
        self.info = info
        self.weight_tensor = weight_tensor
        self.model = HACDEncoder(meta_paths, input_dim, args, num_heads)
        self.opt = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    
    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        self.model.to(device)
        self.model.train()
        total_time_start = time.time()
        for epoch in range(1, self.args.num_epoch):
            time_start = time.time()
            self.opt.zero_grad()
            prob, embedding = self.model(self.g.to(device), self.features.to(device))
            I = torch.eye(self.info.shape[0]).to(device)
            Q = modularity(self.info - I * self.info, prob)
            R = reconstruct(prob, self.info, self.weight_tensor)
            loss = self.args.par1 * Q + self.args.par2 * R
            loss.backward()
            self.opt.step()

            time_end = time.time()
            time_epoch = time_end - time_start
            if self.args.print_yes and epoch % self.args.print_intv == 0:
                print("epoch %d :" % epoch, "time: %f" % time_epoch, "Q is %.10f" % (-1 * Q))

        total_time_end = time.time()
        total_time = total_time_end - total_time_start
        self.model.eval()
        embedding = self.model.embedding(self.g, self.features)
        embedding = embedding.detach().to("cpu")
        sft = nn.Softmax(dim=1)
        prob = sft(embedding)
        modu = compute_Q(torch.Tensor(self.adj.todense()), prob)
        print("The mudularity of HAN: %.4f" % modu)
        print("Total time: %.4f" % total_time)
        return embedding