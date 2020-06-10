# coding=utf-8
# author=yphacker


import numpy as np
import torch


def set_seed():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_selected_text(text, sentiment, start_idx, end_idx, offsets):
    # Set the predicted output as the original tweet when the tweet's sentiment is "neutral",
    # or the tweet only contains one word
    if sentiment == "neutral" or len(text.split()) < 2:
        selected_text = text
        return selected_text
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def calculate_jaccard_score(text, true_selected_text, sentiment, pred_start_idx, pred_end_idx, offsets):
    """
    Calculate the jaccard score from the predicted span and the actual span for a batch of tweets
    """
    # A span's end index has to be greater than or equal to the start index
    # If this doesn't hold, the start index is set to equal the end index (the span is a single token)
    if pred_end_idx < pred_start_idx:
        pred_end_idx = pred_start_idx
    pred_selected_text = get_selected_text(text, sentiment, pred_start_idx, pred_end_idx, offsets)
    score = jaccard(true_selected_text.strip(), pred_selected_text.strip())
    return score


def get_score(y_true_list, y_pred_list):
    jaccard_score = 0
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        text = y_true[0]
        selected_text = y_true[1]
        sentiment = y_true[2]
        offsets = y_true[3]
        pred_start_idx = y_pred[0]
        pred_end_idx = y_pred[1]

        score = calculate_jaccard_score(text, selected_text, sentiment, pred_start_idx, pred_end_idx, offsets)
        jaccard_score += score
    return jaccard_score / len(y_true_list)


set_seed()
