#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 上午9:02
# @Author  : liu yuhan
# @FileName: result_analysis.py
# @Software: PyCharm

import json
from collections import Counter
from parser import *


class DataMaker:
    def __init__(self, args):
        self.link = []
        self.max_len = args.max_len
        with open('../data/111_result.jsonl') as file:
            for l in file:
                d = json.loads(l)
                self.link.append([d['triple_id'], d["result"], d["salience"]])

    def analysis(self):
        """
        这里挑一个最简单的方法
        s(s_class) [不、很]r t(t_class)
        :return:
        """

        result_list = []
        for triple_id, result, label in self.link:
            result_list.append(result)
        result_count = Counter(result_list)
        print(result_count)


if __name__ == '__main__':
    args = parameter_parser()
    mlm = DataMaker(args)
    mlm.analysis()
