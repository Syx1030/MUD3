import torch

import numpy as np
from torch import nn
from tqdm import tqdm

import math
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

cfp = 0.1296
cfn = 1
ctp = 1
ctn = 0
# 真阳第一次发现，假阳概率
first_time_tp = list()
max_tp = list()
max_fp = list()

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def metrics(gold, pred):
    f1, acc, prec, recall = f1_score(gold, pred), accuracy_score(gold, pred), precision_score(gold,
                                                                                              pred), recall_score(
        gold, pred)  # 正样本召回率
    # oss, kind, f1, acc, pre, recall, epoch
    # loss, kind, f1, acc, pre, recall, epoch
    print((f1, acc, prec, recall))
    return f1, acc, prec, recall


def lc0k(target, result, trajectory: list, o=50):
    time, result = 0, result.item()
    for i in range(0, len(trajectory)):
        if trajectory[i] == 1:
            time = 1 - 1 / (1 + math.exp(i - o))
            break
    if result == 0 and target == 0:
        return ctn
    elif result == 1 and target == 1:
        return ctp * time
    elif result == 1 and target == 0:
        return cfp
    elif result == 0 and target == 1:
        return cfn


# F_latency
def f_penalty(k, p):
    return -1 + (2 / (1 + np.exp((-p) * (k))))


def speed(y_pred, y_true, d, p):
    penalty_list = [f_penalty(k=d[i], p=p) for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1]
    print(penalty_list)
    print(np.median(d))
    if len(penalty_list) != 0:
        return 1 - np.median(penalty_list)
    else:
        return 0.


def f_latency(labels, true_labels, delays, penalty=0.037):
    """F Latency performance measure.

    Parameters
    ----------
    labels : numpy.ndarray
        The numpy array of the predicted labels.
    true_labels : list of int
        The list of the true labels.
    delays : numpy.ndarray
        The delays to give a response for every user.
    penalty : float
        The penalty for a delayed classification.

    Returns
    -------
    float
        The F Latency.
    """
    f1 = f1_score(y_pred=labels, y_true=true_labels, average='binary')
    speed_value = speed(y_pred=labels, y_true=true_labels, d=delays, p=penalty)
    print(f1)

    return f1 * speed_value


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data  # 如果不是张量、列表、元组或字典，保持原样



def seq_loss(log_squence_logits,target):
    '''
    :param target:
    :param pred_logits:
    :param log_squence_probs:
    :param choose_posablity: seq_len,2
    :param choose_logits: 1,2
    :param final_logits: 1,2
    :return:
    '''
    # 1. Directly work with log probabilities
    alarm_probs = torch.sigmoid(log_squence_logits)  #每个点的报警概率 batch_size,seq_len,1
    last_probs = torch.sigmoid(log_squence_logits[:,-1,:])
    print(alarm_probs.reshape((-1)))
    not_depressed_probs = 1 - alarm_probs  # 形状为 (batch_size, seq_len, 1)
    #log_alarm_probs = torch.log(alarm_probs + 1e-10)  # 形状为 (batch_size, seq_len, 1)
    log_no_alarm_probs = torch.log(not_depressed_probs + 1e-10)
    log_no_alarm_probs_all = torch.sum(log_no_alarm_probs, dim=1)
    no_result_prob = torch.exp(log_no_alarm_probs_all)
    result_probs = 1-no_result_prob
    print(result_probs)
    loss1 = F.binary_cross_entropy(result_probs, target.reshape_as(result_probs))
    loss2 = F.binary_cross_entropy(last_probs, target.reshape_as(result_probs))
    return loss1+loss2