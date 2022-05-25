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
from tqdm import tqdm


class MLM:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        with open('../data/dev_triple.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d['triple_id'], d["subject"], d["object"], d["predicate"]])
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
        self.model = BertForMaskedLM.from_pretrained(args.pretrain_model_path)

    def predict(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """

        file_write = '../data/000_result.jsonl'

        with open(file_write, 'w', encoding='UTF-8') as json_file:
            for triple_id, s, t, r in tqdm(self.link):
                s_class, r, t_class = r.split('_')
                token_ids_1 = self.tokenizer.encode(s + '(' + s_class + '）')[:-1]
                token_ids_2 = self.tokenizer.encode(r + t + '(' + t_class + '）')[1:]
                inputs = {'input_ids': torch.LongTensor([token_ids_1 + [103] + token_ids_2])}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                mask_token_index = (inputs['input_ids'] == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
                result = self.tokenizer.decode(predicted_token_id)
                if result == '不':
                    json_file.write(json.dumps({'triple_id': triple_id, 'result': result, 'salience': 0}) + '\n')
                else:
                    json_file.write(json.dumps({'triple_id': triple_id, 'result': result, 'salience': 1}) + '\n')


if __name__ == '__main__':
    args = parameter_parser()
    mlm = MLM(args)
    mlm.predict()
