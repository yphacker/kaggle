# coding=utf-8
# author=yphacker


import torch
from torch.utils.data import Dataset
from conf import config


# def process_data(tweet, selected_text, sentiment, tokenizer):
#     """
#     Processes the tweet and outputs the features necessary for model training and inference
#     """
#     len_st = len(selected_text)
#     idx0 = None
#     idx1 = None
#     # Finds the start and end indices of the span
#     for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
#         if tweet[ind: ind + len_st] == selected_text:
#             idx0 = ind
#             idx1 = ind + len_st - 1
#             break
#     # Assign a positive label for the characters within the selected span (based on the start and end indices)
#     char_targets = [0] * len(tweet)
#     if idx0 != None and idx1 != None:
#         for ct in range(idx0, idx1 + 1):
#             char_targets[ct] = 1
#
#     # Encode the tweet using the set tokenizer (converted to ids corresponding to word pieces)
#     tok_tweet = tokenizer.encode(tweet)
#     # Save the ids and offsets (character level indices)
#     # Ommit the first and last tokens, which should be the [CLS] and [SEP] tokens
#     input_ids_orig = tok_tweet.ids[1:-1]
#     tweet_offsets = tok_tweet.offsets[1:-1]
#
#     # A token is considered "positive" when it has at least one character with a positive label
#     # The indices of the "positive" tokens are stored in `target_idx`.
#     target_idx = []
#     for j, (offset1, offset2) in enumerate(tweet_offsets):
#         if sum(char_targets[offset1: offset2]) > 0:
#             target_idx.append(j)
#
#     # Store the indices of the first and last "positive" tokens
#     targets_start = target_idx[0]
#     targets_end = target_idx[-1]
#
#     sentiment_id = {
#         'positive': 3893,
#         'negative': 4997,
#         'neutral': 8699
#     }
#
#     # Prepend the beginning of the tweet with the [CLS] token (101), and tweet sentiment, represented by the `sentiment_id`
#     # The [SEP] token (102) is both prepended and appended to the tweet
#     # You can find the indices in the vocab file: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
#     input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
#     token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
#     mask = [1] * len(token_type_ids)
#     tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
#     targets_start += 3
#     targets_end += 3
#
#     # Pad sequence if its length < `max_len`
#     padding_length = config.max_seq_len - len(input_ids)
#     if padding_length > 0:
#         input_ids = input_ids + ([0] * padding_length)
#         mask = mask + ([0] * padding_length)
#         token_type_ids = token_type_ids + ([0] * padding_length)
#         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
#
#     # Return processed tweet as a dictionary
#     return {
#         'ids': input_ids,
#         'mask': mask,
#         'token_type_ids': token_type_ids,
#         'targets_start': targets_start,
#         'targets_end': targets_end,
#         'orig_tweet': tweet,
#         'orig_selected': selected_text,
#         'sentiment': sentiment,
#         'offsets': tweet_offsets
#     }


# def process_data(tweet, selected_text, sentiment, tokenizer):
#     tweet = " " + " ".join(str(tweet).split())
#     selected_text = " " + " ".join(str(selected_text).split())
#
#     len_st = len(selected_text) - 1
#     idx0 = None
#     idx1 = None
#
#     for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
#         if " " + tweet[ind: ind + len_st] == selected_text:
#             idx0 = ind
#             idx1 = ind + len_st - 1
#             break
#     char_targets = [0] * len(tweet)
#     if idx0 != None and idx1 != None:
#         for ct in range(idx0, idx1 + 1):
#             char_targets[ct] = 1
#
#     tok_tweet = tokenizer.encode(tweet)
#     input_ids_orig = tok_tweet.ids
#     tweet_offsets = tok_tweet.offsets
#
#     target_idx = []
#     for j, (offset1, offset2) in enumerate(tweet_offsets):
#         if sum(char_targets[offset1: offset2]) > 0:
#             target_idx.append(j)
#
#     targets_start = target_idx[0]
#     targets_end = target_idx[-1]
#
#     sentiment_id = {
#         'positive': 1313,
#         'negative': 2430,
#         'neutral': 7974
#     }
#
#     input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
#     token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
#     mask = [1] * len(token_type_ids)
#     tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
#     targets_start += 4
#     targets_end += 4
#
#     padding_length = config.max_seq_len - len(input_ids)
#     if padding_length > 0:
#         input_ids = input_ids + ([1] * padding_length)
#         mask = mask + ([0] * padding_length)
#         token_type_ids = token_type_ids + ([0] * padding_length)
#         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
#
#     return {
#         'ids': input_ids,
#         'mask': mask,
#         'token_type_ids': token_type_ids,
#         'targets_start': targets_start,
#         'targets_end': targets_end,
#         'orig_tweet': tweet,
#         'orig_selected': selected_text,
#         'sentiment': sentiment,
#         'offsets': tweet_offsets
#     }


class MyDataset(Dataset):

    def __init__(self, df, tokenizer, mode='train'):
        self.mode = mode
        self.df = df
        self.tokenizer = tokenizer
        # self.x_data = []
        # self.y_data = []
        # for i, row in df.iterrows():
        #     x, y = self.row_to_tensor(self.tokenizer, row)
        #     self.x_data.append(x)
        #     self.y_data.append(y)
        self.data = []
        for i, row in df.iterrows():
            data = self.row_to_tensor(row)
            try:
                assert len(data["ids"]) == config.max_seq_len
                assert len(data['attention_mask']) == config.max_seq_len
                assert len(data["token_type_ids"]) == config.max_seq_len
                assert len(data["offsets"]) == config.max_seq_len
            except:
                print(data)
                print('ids:{}'.format(len(data["ids"])))
                print('attention_mask:{}'.format(len(data["attention_mask"])))
                print('token_type_ids:{}'.format(len(data["token_type_ids"])))
                print('offsets:{}'.format(len(data["offsets"])))
            # tmp = {
            #     'ids': torch.tensor(data["ids"], dtype=torch.long),
            #     'mask': torch.tensor(data["mask"], dtype=torch.long),
            #     'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            #     'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            #     'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            #     'sentiment': data["sentiment"],
            #     'offsets': torch.tensor(data["offsets"], dtype=torch.long)
            # }
            if self.mode in ['train', 'val']:
                start_idx, end_idx = self.get_target_idx(row, data['tweet'], data['offsets'])
                data['selected_text'] = row['selected_text']
                data['start_idx'] = start_idx
                data['end_idx'] = end_idx
            data['sentiment'] = row['sentiment']
            self.data.append(data)

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind: ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]

        return start_idx, end_idx

    def row_to_tensor(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(encoding.ids) + 1)
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        pad_len = config.max_seq_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            token_type_ids = token_type_ids + ([0] * pad_len)
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids)
        attention_mask = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        token_type_ids = torch.tensor(token_type_ids)
        offsets = torch.tensor(offsets)

        data = dict()
        data['tweet'] = tweet
        data['ids'] = ids
        data['attention_mask'] = attention_mask
        data['token_type_ids'] = token_type_ids
        data['offsets'] = offsets
        return data

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        return self.data[index]

    def __len__(self):
        # return len(self.y_data)
        return len(self.data)
