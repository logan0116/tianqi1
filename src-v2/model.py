#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 下午3:41
# @Author  : liu yuhan
# @FileName: model.py
# @Software: PyCharm


import torch
from transformers import BertModel, BertConfig
from parser import *
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, int(config.vocab_size), bias=False)
        self.bias = nn.Parameter(torch.zeros(int(config.vocab_size)))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MyMLM(nn.Module):
    def __init__(self, args):
        super(MyMLM, self).__init__()
        self.link_size = args.link_size
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        config = BertConfig(hidden_size=self.dim, vocab_size=self.vocab_size)
        # 这里出现一个小问题，config.vocab_size的指向有问题，这边没有从文件载入了，直接写死。
        self.model = BertModel.from_pretrained(args.model_path)
        self.cls = BertOnlyMLMHead(config)
        self.transfer_matrix = self._init_transfer_matrix()
        self.loss = nn.CrossEntropyLoss()

    def _init_transfer_matrix(self):
        transfer_matrix = nn.Embedding(num_embeddings=self.link_size,
                                       embedding_dim=self.dim * self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        transfer_matrix.weight.data.uniform_(-uniform_range, uniform_range)
        return transfer_matrix

    def forward(self, source, attention, r):
        inputs_embeds = self.model(input_ids=source, attention_mask=attention).last_hidden_state
        # mr = self.transfer_matrix(r).view(-1, self.dim, self.dim)
        # h, inputs_embeds, t = inputs_embeds[:, :1, :], inputs_embeds[:, 1:-1, :], inputs_embeds[:, -1:, :]
        # inputs_embeds = torch.matmul(inputs_embeds, mr)
        # inputs_embeds = torch.cat([h, inputs_embeds, t], dim=1)
        inputs_embeds = self.cls(inputs_embeds)
        return inputs_embeds

    def fun_loss(self, inputs_embeds, target):
        return self.loss(inputs_embeds.view(-1, self.vocab_size), target.view(-1))


class Trainer:
    def __init__(self, device, train_loader):
        self.device = device
        self.loader = train_loader
        self.loss_collector = []

    def train(self, epoch, model, optimizer):
        with tqdm(total=len(self.loader)) as bar:
            for source, attention, target, r, mp in self.loader:
                source, attention, target = source.to(self.device), attention.to(self.device), target.to(self.device)
                r = r.to(self.device)
                inputs_embeds = model(source, attention, r)
                loss = model.fun_loss(inputs_embeds, target)
                loss.backward()
                optimizer.step()
                self.loss_collector.append(loss.item())
                bar.set_description('train: Epoch ' + str(epoch))
                bar.set_postfix(loss=loss)
                bar.update(1)
        return model, optimizer, np.mean(self.loss_collector)


class Evaluator:
    def __init__(self, device, test_loader, args):
        self.score = 0
        self.status_best = []
        self.model_save_path = args.model_save_path
        self.device = device
        self.loader = test_loader
        if not os.path.exists(self.model_save_path):
            os.mkdir(self.model_save_path)

    def evaluate(self, epoch, model, loss):
        torch.set_grad_enabled(False)
        tp, fp, tn, fn = 0, 0, 0, 0
        with tqdm(total=len(self.loader)) as bar:
            for source, attention, target, r, mp in self.loader:
                source, attention = source.to(self.device), attention.to(self.device)
                r = r.to(self.device)
                inputs_embeds = model(source, attention, r).cpu()
                predict = inputs_embeds[torch.arange(mp.shape[0]), mp].argmax(axis=1)
                predict = torch.where(predict == 679, 0, torch.ones_like(predict))

                for t, p in zip(target.tolist(), predict.tolist()):
                    if t == 1 and p == 1:
                        tp += 1
                    elif t == 1 and p == 0:
                        fn += 1
                    elif t == 0 and p == 1:
                        fp += 1
                    elif t == 0 and p == 0:
                        tp += 1

                bar.set_description('eval: Epoch ' + str(epoch))
                bar.update(1)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        score = 2 * precision * recall / (precision + recall)
        status = ["epoch", epoch, "loss", loss, 'score', score, 'precision:', precision, 'recall', recall]
        print(status)
        if self.score < score:
            self.score = score
            self.status_best = status
        torch.save(model.state_dict(), self.model_save_path + '/epoch-' + str(epoch))

        torch.set_grad_enabled(True)
