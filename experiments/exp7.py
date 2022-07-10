#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler

from sklearn import model_selection
from sklearn import metrics
import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
#import utils

from pathlib import Path


# In[2]:


class config:
    EXPERIMENT_NAME = "exp7"
    MAX_LEN = 140
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 8
    #GRAD_ACC_STEPS = 2
    EPOCHS = 3 # 5 was useless, earlystopping kicked in
    LEARNING_RATE = 3e-5
    DATA_DIR = Path('')
    MODEL_NAME = "roberta-base"
    TRAINING_FILE = "train_folds.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer( ##explore this
        vocab_file=f"vocab.json", 
        merges_file=f"merges.txt", 
        lowercase=True,
        add_prefix_space=True
    )


# In[3]:


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.MODEL_NAME, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.conv1d_128_0 = nn.Conv1d(768, 128, kernel_size=2,                            stride=1, padding=0, dilation=1, groups=1,                            bias=True, padding_mode='zeros')
        self.conv1d_128_1 = nn.Conv1d(768, 128, kernel_size=2,                            stride=1, padding=0, dilation=1, groups=1,                            bias=True, padding_mode='zeros')
        self.conv1d_64_0 = nn.Conv1d(128, 64, kernel_size=2,                            stride=1, padding=0, dilation=1, groups=1,                            bias=True, padding_mode='zeros')
        self.conv1d_64_1 = nn.Conv1d(128, 64, kernel_size=2,                            stride=1, padding=0, dilation=1, groups=1,                            bias=True, padding_mode='zeros')
        
        self.relu = nn.LeakyReLU()
        
        self.l0 = nn.Linear(64, 1)
        self.l1 = nn.Linear(64, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        
        self.pad = nn.ConstantPad1d((0, 1), 0)
    
    def forward(self, ids, mask, token_type_ids):
        out, _ = self.roberta( #
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        #out =  torch.cat((out[-1], out[-2]), dim=-1)
        out1 = out.permute(0, 2, 1)
        out1 = self.drop_out(out1)
        #print(out.shape)
        out1 = self.pad(out1)
        #print(out.shape)
        out1 = self.conv1d_128_0(out1)
        #print(out.shape)
        out1 = self.relu(out1)
        #print(out.shape)
        out1 = self.pad(out1)
        out1 = self.conv1d_64_0(out1)
        #print(out.shape)
        #out = out.flatten()
        out1 = out1.permute(0, 2, 1)
        #print(out.shape)
        start_logits = self.l0(out1)
        #print(logits.shape)
        
        #out =  torch.cat((out[-1], out[-2]), dim=-1)
        out2 = out.permute(0, 2, 1)
        out2 = self.drop_out(out2)
        #print(out.shape)
        out2 = self.pad(out2)
        #print(out.shape)
        out2 = self.conv1d_128_1(out2)
        #print(out.shape)
        out2 = self.relu(out2)
        #print(out.shape)
        out2 = self.pad(out2)
        out2 = self.conv1d_64_1(out2)
        #print(out.shape)
        #out = out.flatten()
        out2 = out2.permute(0, 2, 1)
        #print(out.shape)
        end_logits = self.l1(out2)
        #print(logits.shape)
        
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# In[4]:


# def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
#     len_st = len(selected_text)
#     idx0 = None
#     idx1 = None
#     for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
#         if tweet[ind: ind+len_st] == selected_text:
#             idx0 = ind
#             idx1 = ind + len_st - 1
#             break

#     char_targets = [0] * len(tweet)
#     if idx0 != None and idx1 != None:
#         for ct in range(idx0, idx1 + 1):
#             char_targets[ct] = 1
    
#     tok_tweet = tokenizer.encode(tweet)
#     input_ids_orig = tok_tweet.ids[1:-1]
#     tweet_offsets = tok_tweet.offsets[1:-1]
    
#     target_idx = []
#     for j, (offset1, offset2) in enumerate(tweet_offsets):
#         if sum(char_targets[offset1: offset2]) > 0:
#             target_idx.append(j)
    
#     targets_start = target_idx[0]
#     targets_end = target_idx[-1]

#     sentiment_id = {
#         'positive': 3893,
#         'negative': 4997,
#         'neutral': 8699
#     }
    
#     input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
#     token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
#     mask = [1] * len(token_type_ids)
#     tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
#     targets_start += 3
#     targets_end += 3

#     padding_length = max_len - len(input_ids)
#     if padding_length > 0:
#         input_ids = input_ids + ([0] * padding_length)
#         mask = mask + ([0] * padding_length)
#         token_type_ids = token_type_ids + ([0] * padding_length)
#         tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
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


# In[5]:


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " " + " ".join(str(selected_text).split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids
    tweet_offsets = tok_tweet.offsets
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 1313,
        'negative': 2430,
        'neutral': 7974
    }
    
    input_ids = [0] + [sentiment_id[sentiment]] + [2] + [2] + input_ids_orig + [2]
    token_type_ids = [0, 0, 0, 0] + [0] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
    targets_start += 4
    targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


# In[6]:


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long)
        }


# In[7]:


def calculate_jaccard_score(
    original_tweet, 
    target_string, 
    sentiment_val, 
    idx_start, 
    idx_end, 
    offsets,
    verbose=False):
    
    if idx_end < idx_start:
        idx_end = idx_start
    
    filtered_output  = ""
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
        if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            filtered_output += " "

    if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


# In[8]:


# def calculate_jaccard_score(
#     original_tweet, 
#     target_string, 
#     sentiment_val, 
#     idx_start, 
#     idx_end, 
#     offsets,
#     verbose=False):
    
#     if idx_end < idx_start:
#         idx_end = idx_start
    
#     filtered_output  = ""
#     for ix in range(idx_start, idx_end + 1):
#         filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
#         if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
#             filtered_output += " "

#     if sentiment_val == "neutral" or len(original_tweet.split()) < 2:
#         filtered_output = original_tweet

#     jac = utils.jaccard(target_string.strip(), filtered_output.strip())
#     return jac, filtered_output


# In[9]:


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# In[10]:


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[11]:


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


# In[12]:


# def eval_fn(data_loader, model, device):
#     model.eval()
#     losses = utils.AverageMeter()
#     jaccards = utils.AverageMeter()
    
#     with torch.no_grad():
#         tk0 = tqdm(data_loader, total=len(data_loader))
#         for bi, d in enumerate(tk0):
#             ids = d["ids"]
#             token_type_ids = d["token_type_ids"]
#             mask = d["mask"]
#             sentiment = d["sentiment"]
#             orig_selected = d["orig_selected"]
#             orig_tweet = d["orig_tweet"]
#             targets_start = d["targets_start"]
#             targets_end = d["targets_end"]
#             offsets = d["offsets"].numpy()

#             ids = ids.to(device, dtype=torch.long)
#             token_type_ids = token_type_ids.to(device, dtype=torch.long)
#             mask = mask.to(device, dtype=torch.long)
#             targets_start = targets_start.to(device, dtype=torch.long)
#             targets_end = targets_end.to(device, dtype=torch.long)

#             outputs_start, outputs_end = model(
#                 ids=ids,
#                 mask=mask,
#                 token_type_ids=token_type_ids
#             )
#             loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
#             outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
#             outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
#             jaccard_scores = []
#             for px, tweet in enumerate(orig_tweet):
#                 selected_tweet = orig_selected[px]
#                 tweet_sentiment = sentiment[px]
#                 jaccard_score, _ = calculate_jaccard_score(
#                     original_tweet=tweet,
#                     target_string=selected_tweet,
#                     sentiment_val=tweet_sentiment,
#                     idx_start=np.argmax(outputs_start[px, :]),
#                     idx_end=np.argmax(outputs_end[px, :]),
#                     offsets=offsets[px]
#                 )
#                 jaccard_scores.append(jaccard_score)

#             jaccards.update(np.mean(jaccard_scores), ids.size(0))
#             losses.update(loss.item(), ids.size(0))
#             tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
#     print(f"Jaccard = {jaccards.avg}")
#     return jaccards.avg


# In[13]:


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


# In[14]:


def dist_between(start_logits, end_logits, device='cuda', max_seq_len=config.MAX_LEN):
    """get dist btw. pred & ground_truth"""

    linear_func = torch.tensor(np.linspace(0, 1, max_seq_len, endpoint=False), requires_grad=False)
    linear_func = linear_func.to(device)

    start_pos = (start_logits*linear_func).sum(axis=1)
    end_pos = (end_logits*linear_func).sum(axis=1)

    diff = end_pos-start_pos

    return diff.sum(axis=0)/diff.size(0)


def dist_loss_fn(start_logits, end_logits, start_positions, end_positions, device='cuda', max_seq_len=config.MAX_LEN, scale=1):
    """calculate distance loss between prediction's length & GT's length
    
    Input
    - start_logits ; shape (batch, max_seq_len{128})
        - logits for start index
    - end_logits
        - logits for end index
    - start_positions ; shape (batch, 1)
        - start index for GT
    - end_positions
        - end index for GT
    """
    start_logits = torch.nn.Softmax(1)(start_logits) # shape ; (batch, max_seq_len)
    end_logits = torch.nn.Softmax(1)(end_logits)
    
    # one hot encoding for GT (start_positions, end_positions)
    start_logits_gt = torch.zeros([len(start_positions), max_seq_len], requires_grad=False).to(device)
    end_logits_gt = torch.zeros([len(end_positions), max_seq_len], requires_grad=False).to(device)
    for idx, _ in enumerate(start_positions):
        _start = start_positions[idx]
        _end = end_positions[idx]
        start_logits_gt[idx][_start] = 1
        end_logits_gt[idx][_end] = 1

    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len)
    gt_dist = dist_between(start_logits_gt, end_logits_gt, device, max_seq_len) # always positive
    diff = (gt_dist-pred_dist)

    rev_diff_squared = 1-torch.sqrt(diff*diff) # as diff is smaller, make it get closer to the one
    loss = -torch.log(rev_diff_squared) # by using negative log function, if argument is near zero -> inifinite, near one -> zero

    return loss*scale


# In[15]:


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"]
        
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)
        
        model.zero_grad()
        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        ) 
        idx_loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        dist_loss = dist_loss_fn(
                outputs_start, outputs_end,
                targets_start, targets_end,
                device) 
        
        loss = (idx_loss + dist_loss)/2
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = calculate_jaccard_score(
                original_tweet=tweet, # Full text of the px'th tweet in the batch
                target_string=selected_tweet, # Span containing the specified sentiment for the px'th tweet in the batch
                sentiment_val=tweet_sentiment, # Sentiment of the px'th tweet in the batch
                idx_start=np.argmax(outputs_start[px, :]), # Predicted start index for the px'th tweet in the batch
                idx_end=np.argmax(outputs_end[px, :]), # Predicted end index for the px'th tweet in the batch
                offsets=offsets[px] # Offsets for each of the tokens for the px'th tweet in the batch
            )
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


# In[16]:


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets = d["offsets"].numpy()

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = calculate_jaccard_score(
                    original_tweet=tweet,
                    target_string=selected_tweet,
                    sentiment_val=tweet_sentiment,
                    idx_start=np.argmax(outputs_start[px, :]),
                    idx_end=np.argmax(outputs_end[px, :]),
                    offsets=offsets[px]
                )
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)
    
    print(f"Jaccard = {jaccards.avg}")
    return jaccards.avg


# In[17]:


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    
    train_dataset = TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")
    model_config = transformers.RobertaConfig.from_pretrained(config.MODEL_NAME)
    #model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    es = EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")
    
    for epoch in range(config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"model_{fold}-{config.EXPERIMENT_NAME}.bin")
        if es.early_stop:
            print("Early stopping")
            break


# In[18]:


run(fold=0)


# In[ ]:


run(fold=1)


# In[ ]:


run(fold=2)


# In[ ]:


run(fold=3)


# In[ ]:


run(fold=4)


# In[ ]:





# # Eval

# In[ ]:


df_test = pd.read_csv("test.csv")
df_test.loc[:, "selected_text"] = df_test.text.values


# In[ ]:


device = torch.device("cuda")
model_config = transformers.RobertaConfig.from_pretrained(config.MODEL_NAME)
######## ????
#model_config.output_hidden_states = True
######## ????


# In[ ]:


model1 = TweetModel(conf=model_config)
model1.to(device)
model1.load_state_dict(torch.load(f"model_0-{config.EXPERIMENT_NAME}.bin"))
model1.eval()

# model2 = TweetModel(conf=model_config)
# model2.to(device)
# model2.load_state_dict(torch.load(f"model_1-{config.EXPERIMENT_NAME}.bin"))
# model2.eval()

# model3 = TweetModel(conf=model_config)
# model3.to(device)
# model3.load_state_dict(torch.load(f"model_2-{config.EXPERIMENT_NAME}.bin"))
# model3.eval()

# model4 = TweetModel(conf=model_config)
# model4.to(device)
# model4.load_state_dict(torch.load(f"model_3-{config.EXPERIMENT_NAME}.bin"))
# model4.eval()

# model5 = TweetModel(conf=model_config)
# model5.to(device)
# model5.load_state_dict(torch.load(f"model_4-{config.EXPERIMENT_NAME}.bin"))
# model5.eval()


# In[ ]:


test_dataset = TweetDataset(
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
    )

data_loader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=config.VALID_BATCH_SIZE,
    num_workers=1
)


# In[ ]:


final_output = []

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"].numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start1, outputs_end1 = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start2, outputs_end2 = model2(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start3, outputs_end3 = model3(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start4, outputs_end4 = model4(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        
        outputs_start5, outputs_end5 = model5(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )
        outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
        outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5
        
#         outputs_start = outputs_start1
#         outputs_end = outputs_end1
        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            _, output_sentence = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            final_output.append(output_sentence)


# In[ ]:


sample = pd.read_csv("sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
sample.to_csv(f"submission-{config.EXPERIMENT_NAME}-5fold.csv", index=False)


# In[ ]:





# In[ ]:


final_output = []

with torch.no_grad():
    tk0 = tqdm(data_loader, total=len(data_loader))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_selected = d["orig_selected"]
        orig_tweet = d["orig_tweet"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        offsets = d["offsets"].numpy()

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start, outputs_end = model1(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        
        outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            _, output_sentence = calculate_jaccard_score(
                original_tweet=tweet,
                target_string=selected_tweet,
                sentiment_val=tweet_sentiment,
                idx_start=np.argmax(outputs_start[px, :]),
                idx_end=np.argmax(outputs_end[px, :]),
                offsets=offsets[px]
            )
            final_output.append(output_sentence)
    
sample = pd.read_csv("sample_submission.csv")
sample.loc[:, 'selected_text'] = final_output
sample.to_csv(f"submission-{config.EXPERIMENT_NAME}-fold1.csv", index=False)

sample.head()


# In[ ]:




