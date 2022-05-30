#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/3 下午9:03
# @Author  : liu yuhan
# @FileName: parser.py
# @Software: PyCharm

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='for cnc ner')
    # model parameter
    parser.add_argument("--model_path", type=str, default="../roberta_chinese")
    parser.add_argument("--pretrain_model_path", type=str, default="../roberta_chinese")
    parser.add_argument("--model_save_path", type=int, default=101)

    parser.add_argument("--link_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=21128)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--cuda_order", type=str, default='0')
    parser.add_argument("--max_len", type=int, default=36)
    # file
    parser.add_argument("--train_file_path", type=str, default="../train.tsv")
    parser.add_argument("--test_file_path", type=str, default="../test.tsv")

    return parser.parse_args()
