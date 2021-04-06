import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
import torch.distributions.cauchy as ca
from numpy import linalg as LA
import random

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) /
                    (1.0 + np.exp(-alpha * iter_num / max_iter)) -
                    (high - low) + low)


def entropy(F1, feat, lamda, eta=1.0):
    out_t1 = F1(feat, reverse=True, eta=-eta)
    out_t1 = F.softmax(out_t1)
    loss = -lamda * torch.mean(torch.sum(out_t1 * (torch.log(out_t1 + 1e-5)), 1))

    return loss

#just trying adding weights to this one for now, then the others
def crossentropy(F1, feat, gt, weights=None):        
    out = F1(feat, donorm=False, reverse=True, eta=-1)

    # 1.cross-entropy loss for classification
    if (weights is not None):
      criterion = nn.CrossEntropyLoss(reduction='none').cuda()
      loss = criterion(out, gt)
      #now multiply the loss and weights together, if weights
      #is too long due to the batch size, just multiply with part of it
      loss = loss*weights[0:loss.shape[0]]
      loss = torch.mean(loss)
    else:
      criterion = nn.CrossEntropyLoss().cuda()
      loss = criterion(out, gt)
    return loss
def focal_loss(F1, feat, gt, weights=None):
    alpha = 0.5
    gamma = 2.0
    logit = F1(feat, donorm=False, reverse=True, eta=-1)
    prob = F.softmax(logit, dim=1) + 1e-6
    gt = gt.view(-1,1)
    onehot = torch.zeros(prob.size()).cuda().scatter_(1, gt,1.0)
    weight = torch.pow(1. - prob, gamma)
    focal = -alpha * weight * torch.log(prob)

    if (weights is not None):
      loss = torch.mean(torch.sum(onehot * focal, dim=1)*weights[0:logit.shape[0]])
    else:
      loss = torch.mean(torch.sum(onehot * focal, dim=1))

    return loss

def asoftmax_loss(F1, feat, gt, weights = None):
    m = 0.8
    logit = F1(feat, donorm=False, reverse=True, eta=-1)
    gt_mod = gt.view(-1,1)
    weight = torch.ones(logit.size()).cuda().scatter_(1, gt_mod, m)
    logit = logit * weight

    if (weights is not None):
      criterion = nn.CrossEntropyLoss(reduction='none').cuda()
      loss = criterion(logit, gt)
      #now multiply the loss and weights together, if weights
      #is too long due to the batch size, just multiply with part of it
      loss = loss*weights[0:loss.shape[0]]
      loss = torch.mean(loss)
    else:
      criterion = nn.CrossEntropyLoss().cuda()
      loss = criterion(logit, gt)

    return loss

def smooth_crossentropy(F1, feat, gt, weights = None):
    logits = F1(feat)
    prob = F.softmax(logits)
    log_prob = -prob.log()
    gt = torch.zeros(prob.size()).cuda().scatter_(1, gt.unsqueeze(1).cuda(), 1)

    epsilon = 0.1
    num_classes = logits.shape[1]
    gt = (1 - epsilon) * gt + epsilon / num_classes
    if (weights is not None):
      loss = torch.mean(torch.sum(gt * log_prob, dim=1)*weights[0:logits.shape[0]])
    else: 
      loss = torch.mean(torch.sum(gt * log_prob, dim=1))
    return loss
