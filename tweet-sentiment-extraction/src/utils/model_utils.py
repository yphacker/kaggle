# coding=utf-8
# author=yphacker

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from conf import config


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Return the sum of the cross entropy losses for both the start and end logits
    """
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


# 有问题
# def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
#     # neg_weight for when pred position < target position
#     # pos_weight for when pred position > target position
#     gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
#     gap = gap.type(torch.float32)
#     return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)
#
#
# def loss_fn_plus(start_logits, end_logits, start_positions, end_positions):
#     loss_fct = nn.CrossEntropyLoss(reduction='none')  # do reduction later
#
#     start_loss = loss_fct(start_logits, start_positions) * pos_weight(start_logits, start_positions, 1, 1)
#     end_loss = loss_fct(end_logits, end_positions) * pos_weight(end_logits, end_positions, 1, 1)
#
#     start_loss = torch.mean(start_loss)
#     end_loss = torch.mean(end_loss)
#
#     total_loss = (start_loss + end_loss)
#     return total_loss


def dist_between(start_logits, end_logits):
    """get dist btw. pred & ground_truth"""
    linear_func = torch.tensor(np.linspace(0, 1, config.max_seq_len, endpoint=False), requires_grad=False).float()
    linear_func = linear_func.to(config.device)

    # 版本bug, 相乘需要类型一致
    start_pos = (start_logits * linear_func).sum(axis=1)
    end_pos = (end_logits * linear_func).sum(axis=1)

    diff = end_pos - start_pos

    return diff.sum(axis=0) / diff.size(0)


def calc_dist_loss(start_logits, end_logits, start_positions, end_positions, scale=1):
    # calculate distance loss between prediction's length & GT's length
    start_logits = torch.nn.Softmax(1)(start_logits)
    end_logits = torch.nn.Softmax(1)(end_logits)

    start_one_hot = one_hot(start_positions, num_classes=config.max_seq_len).to(config.device).float()
    end_one_hot = one_hot(end_positions, num_classes=config.max_seq_len).to(config.device).float()
    pred_dist = dist_between(start_logits, end_logits)
    gt_dist = dist_between(start_one_hot, end_one_hot)  # always positive
    diff = (gt_dist - pred_dist)
    # as diff is smaller, make it get closer to the one
    rev_diff_squared = 1 - torch.sqrt(diff * diff)
    # by using negative log function, if argument is near zero -> inifinite, near one -> zero
    loss = -torch.log(rev_diff_squared)

    return loss * scale


def dist_loss_fn(start_logits, end_logits, start_positions, end_positions):
    start_loss = torch.nn.CrossEntropyLoss()(start_logits, start_positions)
    end_loss = torch.nn.CrossEntropyLoss()(end_logits, end_positions)

    idx_loss = (start_loss + end_loss)

    dist_loss = calc_dist_loss(
        start_logits, end_logits,
        start_positions, end_positions)

    total_loss = idx_loss + dist_loss
    return total_loss


if __name__ == '__main__':
    # argmax pred for the start is 4, target is 1
    start = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
    start_target = torch.Tensor([1]).long()

    # argmax pred for the end is 3, target is 3
    end = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
    end_target = torch.Tensor([3]).long()
    # x = loss_fn_plus(start, end, start_target, end_target)
    x = dist_loss_fn(start, end, start_target, end_target)

    # argmax pred for the start is 2, target is 1
    # argmax pred for the end is 3, target is 3
    start = torch.Tensor([[0.1, 0.1, 0.8, 0.1, 0.1]]).float()
    start_target = torch.Tensor([1]).long()

    end = torch.Tensor([[0.1, 0.1, 0.1, 0.8, 0.1]]).float()
    end_target = torch.Tensor([3]).long()
    # y = loss_fn_plus(start, end, start_target, end_target)
    y = dist_loss_fn(start, end, start_target, end_target)
    # https://www.kaggle.com/jeinsong/distance-loss?scriptVersionId=33216470
    # print(x, y)
