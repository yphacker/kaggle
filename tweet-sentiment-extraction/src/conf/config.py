# coding=utf-8
# author=yphacker

import os
import torch

work_path = os.path.dirname(os.path.abspath('.'))
input_path = os.path.join(work_path, "input")
data_path = os.path.join(input_path, "tweet-sentiment-extraction")
train_path = os.path.join(data_path, 'train.csv')
test_path = os.path.join(data_path, 'test.csv')
sample_submission_path = os.path.join(data_path, 'sample_submission.csv')
model_path = os.path.join(work_path, "model")
for path in [model_path]:
    if not os.path.isdir(path):
        os.makedirs(path)
#     暂时没有截取，有很多符号的，长度不止110
#     max_seq_len = 110
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_seq_len = 128
n_splits = 5
patience_epoch = 2

batch_size = 32
epochs_num = 1
train_print_step = 100
