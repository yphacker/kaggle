# coding=utf-8
# author=yphacker


import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from conf import model_config_roberta as model_config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = RobertaConfig.from_pretrained(model_config.pretrain_model_path, output_hidden_states=True)
        self.model = RobertaModel.from_pretrained(model_config.pretrain_model_path, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # self.classifier = nn.Linear(self.config.hidden_size * 2, 2)
        self.classifier = nn.Linear(self.config.hidden_size, 2)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_states = outputs[2]
        # out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
        out = torch.stack([hidden_states[-1], hidden_states[-2], hidden_states[-3]])
        out = torch.mean(out, 0)
        out = self.dropout(out)
        logits = self.classifier(out)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
