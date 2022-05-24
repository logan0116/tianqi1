#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 上午9:04
# @Author  : liu yuhan
# @FileName: dataloader.py
# @Software: PyCharm

from transformers import BertForMaskedLM, BertTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
from parser import *
import torch
import json


class DataMaker:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        with open('../data/train_triple.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d["subject"], d["object"], d["predicate"], d['salience']])
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)

    def data_maker(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """

        source_list, attention_list, target_list = [], [], []

        label_dict = {'0': 679,
                      '1': 2523}
        for s, t, r, label in self.link:
            s_class, r, t_class = r.split('_')
            token_ids_1 = self.tokenizer.encode(s + '(' + s_class + '）', truncation=True)[:-1]
            token_ids_2 = self.tokenizer.encode(r + t + '(' + t_class + '）', truncation=True)[1:]

            source = token_ids_1 + [103] + token_ids_2
            attention = [1] * len(source)
            target = [-100] * len(token_ids_1) + [label_dict[label]] + [-100] * len(token_ids_2)

            if len(source) <= self.max_len:
                source = source + [0] * (self.max_len - len(source))
                attention = attention + [0] * (self.max_len - len(attention))
                target = target + [0] * (self.max_len - len(target))

            source_list.append(source)
            attention_list.append(attention)
            target_list.append(target)

        return {'input_ids': torch.LongTensor(source_list),
                'labels': torch.LongTensor(target_list)}


if __name__ == '__main__':
    args = parameter_parser()
    print('data loading...')
    data_maker = DataMaker(args)
    data_train = data_maker.data_maker()
    data_train = Dataset.from_dict(data_train)
    print('    data load done.')

    # cuda
    device = torch.device("cuda:" + str(args.cuda_order) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    model = BertForMaskedLM.from_pretrained(args.model_path)
    print('    No of parameters: ', model.num_parameters())
    print('    model load done.')

    training_args = TrainingArguments(
        output_dir='../data/outputs/',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        save_steps=10000,
        do_train=True,
        prediction_loss_only=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
    )
    # Training
    if training_args.do_train:
        checkpoint = None
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
