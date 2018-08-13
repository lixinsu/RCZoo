#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
from collections import defaultdict


def merge_answers_max(answers):
    return sorted(answers, key=lambda x:x[1], reverse=True)[0]


def get_final_answer(infile, outfile, merge_func=merge_answers_max):
    """
    Merge answers extracted from multiple QP pair for one query


    """
    qid2pred = json.load(open(infile))
    real_qid2answers = defaultdict(list)
    real_qid2answer = {}
    for k,v in qid2pred.items():
        real_k = k.split('-')[0]
        real_qid2answers[real_k].append(v[0])
    for k,v in real_qid2answers.items():
        real_qid2answer[k] = merge_func(v)
    json.dump(real_qid2answer, open(outfile, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    get_final_answer(sys.argv[1], sys.argv[1].replace('preds','real_preds'))

