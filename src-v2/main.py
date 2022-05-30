#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 上午9:04
# @Author  : liu yuhan
# @FileName: main.py
# @Software: PyCharm

from parser import *
import torch.utils.data as Data
import torch.optim as optim

from utils import *
from model import *

import sys

sys.setrecursionlimit(30000)


def train():
    """
    直接进行一波训练
    :return:
    """
    args = parameter_parser()
    print('data loading...')
    data_maker = DataMaker(args)
    # 数据集分割
    train_set, test_set = data_maker.data_split()
    # 训练集
    source_list, attention_list, target_list, r_list = data_maker.data_maker(train_set)
    train_loader = Data.DataLoader(MyDataSet(source_list, attention_list, target_list, r_list), args.batch_size, True)
    # 验证集
    source_list, attention_list, target_list, r_list = data_maker.data_maker(test_set)
    test_loader = Data.DataLoader(MyDataSet(source_list, attention_list, target_list, r_list), args.batch_size, True)
    # cuda
    device = torch.device("cuda:" + str(args.cuda_order) if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # 模型初始化
    print('model loading...')
    print('    model:', args.model_path)
    model = MyMLM(args)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print('    model load done.')

    print('training...')
    for epoch in range(args.epochs):
        loss_collector = []
        with tqdm(total=len(train_loader)) as bar:
            for source, attention, target, r in train_loader:
                source, attention, target, r = source.to(device), attention.to(device), target.to(device), r.to(device)
                inputs_embeds = model(source, attention, r)
                loss = model.loss(inputs_embeds, target)
                loss.backward()
                optimizer.step()
                loss_collector.append(loss.item())
                bar.set_description('train: Epoch ' + str(epoch))
                bar.set_postfix(loss=loss)
                bar.update(1)
    #     if epoch % 10 == 0 or epoch > 0:
    # #         evaluator.evaluate(epoch, trans_model, test_list, np.mean(loss_collector))
    # #
    # # print(evaluator.status_best)


if __name__ == '__main__':
    train()
