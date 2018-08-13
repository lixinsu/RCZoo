#!/usr/bin/env python
# coding: utf-8

import sys
import json


def evaluate(pred_file, origin_file):
    qid2answer = {}
    qid2pred = json.load(open(pred_file))
    for line in open(origin_file):
        data = json.loads(line)
        qid2answer[ str(data['query_id']) ] = '' if (data['answer'] == 'x' or data['answer'] == '0') else data['answer']
    nb_answerable  = len([x for x in qid2answer if qid2answer[x] != ''])
    print(len(qid2pred), len(qid2answer))
    print( type(list(qid2answer.keys())[0]) )
    qid2em = {}
    for qid in qid2pred:
        if qid2pred[qid][0] == qid2answer[qid]:
            qid2em[qid] = 1
        else:
            print(qid2pred[qid][0], qid2answer[qid])
            qid2em[qid] = 0
    res = []
    for qid in qid2pred:
        res.append([qid2em[qid], qid2pred[qid][1]])
    res.sort(key=lambda x:x[1], reverse=True)
    print(res)
    precision_numerator, precision_denominator = 0.0, 0.0
    print('total right answer %s' % sum(qid2em.values()))
    for x in res:
        precision_numerator += x[0]
        precision_denominator += 1
        precision = precision_numerator/precision_denominator
        if precision <= 0.8:
            print(precision_numerator, precision_denominator, nb_answerable)
            print('precision %.4f' % precision)
            print('recall %.4f' % (precision_numerator/ nb_answerable) )
            break


if __name__ == '__main__':
    evaluate(sys.argv[1], sys.argv[2])
