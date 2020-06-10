# coding=utf-8
# author=yphacker

import os
from conf import config
from tokenizers import BertWordPieceTokenizer

model_name = 'bert'
pretrain_model_name = 'bert-base-uncased'
pretrain_model_path = os.path.join(config.input_path, pretrain_model_name)
tokenizer = BertWordPieceTokenizer('{}/vocab.txt'.format(pretrain_model_path), lowercase=True)
learning_rate = 5e-5
adjust_lr_num = 0