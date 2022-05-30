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
        mr = self.transfer_matrix(r).view(-1, self.dim, self.dim)
        h, inputs_embeds, t = inputs_embeds[:, :1, :], inputs_embeds[:, 1:-1, :], inputs_embeds[:, -1:, :]
        inputs_embeds = torch.matmul(inputs_embeds, mr)
        inputs_embeds = torch.cat([h, inputs_embeds, t], dim=1)
        inputs_embeds = self.cls(inputs_embeds)
        return inputs_embeds

    def loss(self, inputs_embeds, target):
        return self.loss(inputs_embeds.view(-1, self.vocab_size), target.view(-1))
