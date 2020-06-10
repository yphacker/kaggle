# coding=utf-8
# author=yphacker

import os
import gc
import time
import argparse
import pandas as pd
from tqdm import tqdm
from importlib import import_module
import torch
import torch.nn as nn
from ranger import Ranger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
# from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from conf import config
from utils.data_utils import MyDataset
# from utils.model_utils import loss_fn
# from utils.model_utils import loss_fn_plus as loss_fn
from utils.model_utils import dist_loss_fn as loss_fn
from utils.utils import set_seed, get_selected_text, get_score


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def get_inputs(batch_x, batch_y=None):
#     if model_name in ['bert', "roberta", 'xlmroberta']:
#         batch_x = tuple(t.to(device) for t in batch_x)
#         inputs = dict(input_ids=batch_x[0], attention_mask=batch_x[1])
#         if model_name in ["bert"]:
#             inputs['token_type_ids'] = batch_x[2]
#         return inputs
#     else:
#         return dict(input_ids=batch_x.to(device))


def evaluate(model, val_loader):
    model.eval()
    data_len = 0
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for inputs in val_loader:
            batch_len = len(inputs)
            data_len += batch_len
            tweet = inputs["tweet"]
            selected_text = inputs["selected_text"]
            sentiment = inputs["sentiment"]
            ids = inputs["ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            targets_start_idx = inputs["start_idx"]
            targets_end_idx = inputs["end_idx"]
            offsets = inputs["offsets"]
            # Move ids, masks, and targets to gpu while setting as torch.long
            ids = ids.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
            attention_mask = attention_mask.to(config.device, dtype=torch.long)
            targets_start_idx = targets_start_idx.to(config.device, dtype=torch.long)
            targets_end_idx = targets_end_idx.to(config.device, dtype=torch.long)
            start_logits, end_logits = model(
                input_ids=ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, targets_start_idx, targets_end_idx)
            total_loss += loss.item() * batch_len
            pred_start_idxs = torch.argmax(start_logits, dim=1).cpu().data.numpy()
            pred_end_idxs = torch.argmax(end_logits, dim=1).cpu().data.numpy()
            for i in range(len(tweet)):
                y_true_list.append((tweet[i], selected_text[i], sentiment[i], offsets[i]))
                y_pred_list.append((pred_start_idxs[i], pred_end_idxs[i]))
    return total_loss / data_len, get_score(y_true_list, y_pred_list)


def train(train_data, val_data, fold_idx=None):
    train_dataset = MyDataset(train_data, tokenizer)
    val_dataset = MyDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = Model().to(config.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    optimizer = Ranger(model.parameters(), lr=5e-5)
    period = int(len(train_loader) / config.train_print_step)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=period, eta_min=5e-9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_print_step, eta_min=1e-9)

    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.model_path, '{}_fold{}.bin'.format(model_name, fold_idx))

    best_val_score = 0
    last_improved_epoch = 0
    adjust_lr_num = 0
    y_true_list = []
    y_pred_list = []
    for cur_epoch in range(config.epochs_num):
        start_time = int(time.time())
        model.train()
        print('epoch:{}, step:{}'.format(cur_epoch + 1, len(train_loader)))
        cur_step = 0
        # for batch_x, batch_y in train_loader:
        for inputs in train_loader:
            tweet = inputs["tweet"]
            selected_text = inputs["selected_text"]
            sentiment = inputs["sentiment"]
            ids = inputs["ids"]
            attention_mask = inputs["attention_mask"]
            token_type_ids = inputs["token_type_ids"]
            targets_start_idx = inputs["start_idx"]
            targets_end_idx = inputs["end_idx"]
            offsets = inputs["offsets"]
            # Move ids, masks, and targets to gpu while setting as torch.long
            ids = ids.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
            attention_mask = attention_mask.to(config.device, dtype=torch.long)
            targets_start_idx = targets_start_idx.to(config.device, dtype=torch.long)
            targets_end_idx = targets_end_idx.to(config.device, dtype=torch.long)
            optimizer.zero_grad()
            start_logits, end_logits = model(
                input_ids=ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )
            loss = loss_fn(start_logits, end_logits, targets_start_idx, targets_end_idx)
            loss.backward()
            optimizer.step()
            cur_step += 1
            # pred_start_idxs = torch.argmax(start_logits, dim=1).cpu().data.numpy()
            # pred_end_idxs = torch.argmax(end_logits, dim=1).cpu().data.numpy()
            # for i in range(len(tweet)):
            #     y_true_list.append((tweet[i], selected_text[i], sentiment[i], offsets[i]))
            #     y_pred_list.append((pred_start_idxs[i], pred_end_idxs[i]))
            if cur_step % config.train_print_step == 0:
                scheduler.step()
                msg = 'the current step: {0}/{1}, cost: {2}s'
                print(msg.format(cur_step, len(train_loader), int(time.time()) - start_time))
            #     train_score = get_score(y_true_list, y_pred_list)
            #     msg = 'the current step: {0}/{1}, train score: {2:>6.2%}'
            #     print(msg.format(cur_step, len(train_loader), train_score))
            #     y_true_list = []
            #     y_pred_list = []
        val_loss, val_score = evaluate(model, val_loader)
        if val_score >= best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val loss: {2:>5.2}, val acc: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.epochs_num, val_loss, val_score,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch >= config.patience_epoch:
            if adjust_lr_num >= model_config.adjust_lr_num:
                print("No optimization for a long time, auto stopping...")
                break
            print("No optimization for a long time, adjust lr...")
            last_improved_epoch = cur_epoch  # 加上，不然会连续更新的
            adjust_lr_num += 1

    del model
    gc.collect()

    if fold_idx is not None:
        model_score[fold_idx] = best_val_score


def predict():
    model = Model().to(config.device)
    model_save_path = os.path.join(config.model_path, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))

    test_df = pd.read_csv(config.test_path)
    test_df.loc[:, 'selected_text'] = test_df.text.values
    submission = pd.read_csv(config.sample_submission_path)

    test_dataset = MyDataset(test_df, tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    pred_list = []
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            tweet = inputs["tweet"]
            sentiment = inputs["sentiment"]
            ids = inputs["ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            offsets = inputs["offsets"]
            ids = ids.to(config.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(config.device, dtype=torch.long)
            mask = mask.to(config.device, dtype=torch.long)
            start_logits, end_logits = model(
                input_ids=ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            )
            pred_start_idx = torch.argmax(start_logits, dim=1).cpu().data.numpy()
            pred_end_idx = torch.argmax(end_logits, dim=1).cpu().data.numpy()

            for i in range(len(tweet)):
                pred = get_selected_text(tweet[i], sentiment[i], pred_start_idx[i], pred_end_idx[i], offsets[i])
                pred_list.append(pred)
    submission['selected_text'] = pred_list
    submission.to_csv('submission.csv', index=False)


def main(op):
    if op == 'train':
        train_df = pd.read_csv(config.train_path)
        train_df = train_df[pd.notnull(train_df['text'])]
        # train_df = train_df[:100]
        if args.mode == 1:
            x = train_df['text'].values
            y = train_df['sentiment'].values
            skf = StratifiedKFold(n_splits=config.n_splits, random_state=0, shuffle=True)
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x, y)):
                train(train_df.iloc[train_idx], train_df.iloc[val_idx], fold_idx)
            score = 0
            score_list = []
            for fold_idx in range(config.n_splits):
                score += model_score[fold_idx]
                score_list.append('{:.4f}'.format(model_score[fold_idx]))
            print('val score:{}, avg val score:{:.4f}'.format(','.join(score_list), score / config.n_splits))
        else:
            train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=0, shuffle=True)
            print('train:{}, val:{}'.format(train_data.shape[0], val_data.shape[0]))
            train(train_data, val_data)
    elif op == 'eval':
        pass
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    set_seed()
    parser = argparse.ArgumentParser(description='tweet-sentiment-extraction')
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--epochs_num", default=8, type=int, help="train epochs")
    parser.add_argument("-m", "--model", default='bert', type=str, required=True,
                        help="choose a model: bert")
    parser.add_argument("-mode", "--mode", default=1, type=int, help="train mode")
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epochs_num = args.epochs_num
    model_name = args.model
    model_file = import_module('models.{}'.format(model_name))
    Model = model_file.Model
    model_config = import_module('conf.model_config_{}'.format(model_name))
    tokenizer = model_config.tokenizer

    model_score = dict()
    main(args.operation)
