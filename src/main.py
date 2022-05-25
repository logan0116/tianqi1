#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 上午9:04
# @Author  : liu yuhan
# @FileName: main.py
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
        self.prompt_size = 2
        with open('../data/train_triple.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d["subject"], d["object"], d["predicate"], d['salience']])
        self.tokenizer = BertTokenizer.from_pretrained(args.model_path)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.1)

    def data_maker(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """
        p_1 = [i for i in range(1, 1 + self.prompt_size)]
        p_2 = [i for i in range(self.prompt_size + 1, self.prompt_size * 2 + 1)]

        source_list, attention_list, target_list = [], [], []

        label_dict = {'0': 679,
                      '1': 2523}
        for s, t, r, label in self.link:
            s_c, r, t_c = r.split('_')
            token_ids_1 = self.tokenizer.encode(s + '(' + s_c + '）', truncation=True, max_length=self.max_len)[:-1]
            token_ids_2 = self.tokenizer.encode(r + t + '(' + t_c + '）', truncation=True, max_length=self.max_len)[1:]

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

        return {'input_ids': torch.stack(source_list),
                'labels': torch.stack(target_list)}


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
