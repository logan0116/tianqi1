#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 上午9:33
# @Author  : liu yuhan
# @FileName: test.py
# @Software: PyCharm

import torch
from torch.nn import CrossEntropyLoss

predict = torch.LongTensor([679, 0, 0])
predict = torch.where(predict == 679, 0, torch.ones_like(predict))

print(predict)
