#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 下午3:39
# @Author  : liu yuhan
# @FileName: utils.py
# @Software: PyCharm


from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
from parser import *
from tqdm import tqdm
import torch
import json
import torch.utils.data as Data


class DataMaker:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        self.prompt_size = 2
        with open('../data/train_triple.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d["subject"], d["object"], d["predicate"], d['salience']])
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.1)

    def data_split(self, rate=0.8):
        """
        拆分训练集和验证集
        """
        data_size = len(self.link)
        train_size = int(rate * data_size)
        test_size = data_size - train_size
        torch.manual_seed(0)
        train_list, test_list = Data.random_split(
            dataset=self.link,
            lengths=[train_size, test_size],
        )
        return train_list, test_list

    def data_maker(self, link_list):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """
        p_1 = [i for i in range(1, 1 + self.prompt_size)]
        p_2 = [i for i in range(self.prompt_size + 1, self.prompt_size * 2 + 1)]

        source_list, attention_list, target_list, r_list = [], [], [], []

        label_dict = {'0': 679,
                      '1': 2523}
        # 这里增加一个类的信息
        relation_dict = {'品类_适用_场景': 0,
                         '品类_适用_人物': 1,
                         '品类_搭配_品类': 2,
                         '人物_蕴含_场景': 3}
        for s, t, r0, label in tqdm(link_list):
            s_c, r, t_c = r0.split('_')
            token_ids_1 = self.tokenizer.encode(s + '(' + s_c + '）', truncation=True, max_length=self.max_len)[:-1]
            token_ids_2 = self.tokenizer.encode(r + t + '(' + t_c + '）', truncation=True, max_length=self.max_len)[1:]
            # 这里做一个简单的mask
            token_ids_1_masked = self.data_collator([token_ids_1])['input_ids'][0].tolist()
            token_ids_2_masked = self.data_collator([token_ids_2])['input_ids'][0].tolist()

            source = torch.LongTensor(token_ids_1_masked + p_1 + [103] + p_2 + token_ids_2_masked)
            attention = torch.ones_like(source)
            target = torch.LongTensor(token_ids_1 + p_1 + [label_dict[label]] + p_2 + token_ids_2)
            target = torch.where(source == self.tokenizer.mask_token_id, target, -100)

            if source.shape[0] < self.max_len:
                zero_pad = torch.zeros(self.max_len - source.shape[0], dtype=torch.int)
                source = torch.cat((source, zero_pad), 0)
                attention = torch.cat((attention, zero_pad), 0)
                target = torch.cat((target, zero_pad), 0)

            source_list.append(source)
            attention_list.append(attention)
            target_list.append(target)
            r_list.append(relation_dict[r0])

        return torch.stack(source_list), torch.stack(attention_list), torch.stack(target_list), torch.LongTensor(r_list)


class MyDataSet(Data.Dataset):
    '''
    没啥说的，正常的数据载入
    '''

    def __init__(self, source_list, attention_list, target_list, r_list):
        self.source_list = source_list
        self.attention_list = attention_list
        self.target_list = target_list
        self.r_list = r_list

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, idx):
        return self.source_list[idx], self.attention_list[idx], self.target_list[idx], self.r_list[idx],
