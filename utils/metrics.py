import numpy as np
from munkres import Munkres
import torch

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
'''def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if np.any(l2 == i):
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro'''

def cluster_acc(y_true, y_pred):
    l1 = list(set(y_true))
    l2 = list(set(y_pred))

    # 创建成本矩阵
    cost_matrix = np.zeros((len(l1), len(l2)))
    for i, true_label in enumerate(l1):
        for j, pred_label in enumerate(l2):
            # 成本为不匹配的数量
            cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred != pred_label))

    # 使用 linear_sum_assignment 找到最小成本的匹配
    row_ind, col_ind = linear(cost_matrix)

    # 创建新的预测结果，根据找到的匹配重新标记
    new_pred = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        new_pred[y_pred == l2[j]] = l1[i]

    # 计算准确率和 F1 分数
    acc = accuracy_score(y_true, new_pred)
    f1 = f1_score(y_true, new_pred, average='macro')  # 或者使用 'micro', 'weighted' 等

    return acc, f1

def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    #print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1

