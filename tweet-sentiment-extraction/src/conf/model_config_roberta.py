# coding=utf-8
# author=yphacker


import os
from conf import config
from tokenizers import ByteLevelBPETokenizer

model_name = 'roberta'
# pretrain_model_name = 'roberta-base'
pretrain_model_name = 'roberta-large'
pretrain_model_path = os.path.join(config.input_path, pretrain_model_name)
tokenizer = ByteLevelBPETokenizer(
    vocab_file='{}/vocab.json'.format(pretrain_model_path),
    merges_file='{}/merges.txt'.format(pretrain_model_path),
    lowercase=True,
    add_prefix_space=True
)
learning_rate = 5e-5
adjust_lr_num = 0
