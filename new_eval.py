#!/usr/bin/env python
# coding: utf-8



import json
import sys
import os


def load_pred(filename):
    ans = json.load(open(filename))
    print('prediction numbers: %s ' % len(ans))
    return ans


def load_gt(filename):
    ans = {}
    with open(filename) as f:
        for l in f:
            d = json.loads(l)
            ans[d['query_id']] = d['answers']
    return ans


def lengths(gts):
    return sum([len(x) for x in gts]) * 1.0 / len(gts)


def validate(pred_ans, gt_ans, ans_span=0, ans_len=[0,5]):
    ems = []
    f1s = []
    for qid in pred_ans:
        pred = pred_ans[qid]
        gts = gt_ans[qid]
        #print(pred)
        #print(gts)
        if len(gts) != ans_span:
            if ans_span == 0 :
                pass
            elif ans_span == 5 and len(gts) > 5:
                pass
            else:
                continue
        if lengths(gts) > ans_len[0] and lengths(gts) <= ans_len[1]:
            pass
        else:
            continue
        TP = set(pred) & set(gts)
        if set(pred) == set(gts):
            ems.append(1)
        else:
            ems.append(0)
        if len(pred) == 0:
            p = 0
            r = 0
            f1 = 0
        else:
            p = len(TP) * 1.0 / len(pred)
            r = len(TP) * 1.0 / len(gts)
            if p == 0 and  r == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p+r)
        f1s.append(f1)
    f1 = sum(f1s) /len(f1s)
    em = sum(ems) *1.0 / len(ems)
    print('%s\t%s\t%s\t%s' % (ans_len, em,f1,len(f1s)))
    return f1, em


if __name__ == "__main__":
    pred_ans = load_pred(sys.argv[1])
    gt_ans = load_gt(sys.argv[2])
    #for i in range(0, 6):
    #    validate(pred_ans, gt_ans, ans_span=i)
    #for ans_len in [[0,5],[5,7],[7,10],[10,15], [15,20]]:
    #    validate(pred_ans, gt_ans, ans_span=0, ans_len=ans_len)
    validate(pred_ans, gt_ans, ans_span=0, ans_len=[0,20])
