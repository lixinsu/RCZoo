#!/usr/bin/env python
# coding: utf-8

import sys
import json
import codecs
import os.path as osp
import os

def strQ2B(ustring):
    """
    python3 全角转半角
    """
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def process_train(filename, outfile):
    fo = codecs.open(outfile, 'w', 'utf8')
    skip_count = 0
    err_json = 0
    total = 0
    arab2chn = {'1': '一', '2': '二', '3': "三", '4':'四', '5':'五', '6':'六', '7':'七', '8':'八', '9':'九'}
    with codecs.open(filename, 'r', 'utf8') as f:
        for l in f:
            if not l.strip():
                continue
            ret = {}
            try:
                data = json.loads(l)
            except:
                err_json += 1
                continue
            ret['query'] = strQ2B(data['query'])
            ps = data['passages']
            ans = strQ2B(data['answer'])
            for idx, p in enumerate(ps):
                p = strQ2B(p['passage_text'])
                ret['answers'] = []
                if ans in p:
                    ret['query_id'] = str(data['query_id']) + '-' + str(idx)
                    ans1 = {'answer_start':p.find(ans), 'text':ans}
                    ret['answers'].append(ans1)
                    ret['passage'] = p
                    fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
                    total += 1

            if 'passage' not in ret:
                #  patch strategy for more data
                ans_chn = ''
                for c in ans:
                    ans_chn += arab2chn.get(c, c)
                ans = ans_chn
                for idx, p in enumerate(ps):
                    p = strQ2B(p['passage_text'])
                    ret['answers'] = []
                    if ans in p:
                        ret['query_id'] = str(data['query_id']) + '-' + str(idx)
                        ans1 = {'answer_start':p.find(ans), 'text':ans}
                        ret['answers'].append(ans1)
                        ret['passage'] = p
                        fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
                        total += 1

                if 'passage' not in ret:
                    skip_count += 1

    print('total err_json %s lines' % err_json)
    print("total skip %s lines" % skip_count)
    print("remain %s lines "  % total)

def process_test(filename, outfile):
    """
    Convert mp-q to multiple q-p pair for reading comprehension model
    """
    fo = codecs.open(outfile, 'w', 'utf8')
    err_json = 0
    total = 0
    with codecs.open(filename, 'r', 'utf8') as f:
        for l in f:
            if not l.strip():
                continue
            ret = {}
            try:
                data = json.loads(l)
            except:
                err_json += 1
                continue
            ret['query'] = strQ2B(data['query'])
            ps = data['passages']
            ans = strQ2B(data['answer'])
            for idx, p in enumerate(ps):
                ret['answers'] = []
                ret['query_id'] = str(data['query_id']) + '-' + str(idx)
                p = strQ2B(p['passage_text'])
                if len(p) < 1:
                    continue
                p = p[:1000]
                ans1 = {'answer_start':p.find(ans), 'text':ans}
                ret['answers'].append(ans1)
                ret['passage'] = p
                total += 1
                fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
    print('total err_json %s lines' % err_json)
    print("remain %s lines "  % total)

if __name__ == "__main__":
    if sys.argv[3] == "train":
        process_train(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == "test":
        process_test(sys.argv[1], sys.argv[2])
