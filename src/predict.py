#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 下午4:28
# @Author  : liu yuhan
# @FileName: predict.py
# @Software: PyCharm


from transformers import BertTokenizer, BertForMaskedLM
import torch
import json
from parser import *


class DataMaker:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        with open('../data/dev_triple.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d['triple_id'], d["subject"], d["object"], d["predicate"]])
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)

    def data_maker(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """
        source_list, target_list = [], []
        label_dict = {'0': 679,
                      '1': 2523}
        for triple_id, s, t, r in self.link:
            s_class, r, t_class = r.split('_')
            token_ids_1 = self.tokenizer.encode(s + '(' + s_class + '）', truncation=True)[:-1]
            token_ids_2 = self.tokenizer.encode(r + t + '(' + t_class + '）', truncation=True)[1:]

            source = token_ids_1 + [103] + token_ids_2

            if len(source) <= self.max_len:
                source = source + [0] * (self.max_len - len(source))
                target = target + [0] * (self.max_len - len(target))

            source_list.append(source)
            target_list.append(target)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}


if __name__ == '__main__':
    args = parameter_parser()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = BertForMaskedLM.from_pretrained(args.model_path)

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    tokenizer.decode(predicted_token_id)

    labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    # mask labels of non-[MASK] tokens
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    print(labels)

    outputs = model(**inputs, labels=labels)
    round(outputs.loss.item(), 2)
