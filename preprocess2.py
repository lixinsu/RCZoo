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
    url_stat = {}
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
            ret['query_id'] = str(data['query_id'])
            ret['query'] = strQ2B(data['query'])
            ret['answers'] = []
            ps = data['passages']
            ans = strQ2B(data['answer']).replace(' ', '')
            allp = set()
            for p in ps:
                u = p['url'].split('/')[2]
                p = strQ2B(p['passage_text'])

                if p in allp:
                    continue
                allp.add(p)

                ret['answers'] = []
                if u not in url_stat:
                    url_stat[u] = [0,0,0.0]
                if ans in p:
                    ans1 = {'answer_start':p.find(ans), 'text':ans}
                    ret['answers'].append(ans1)
                    ret['passage'] = p
                    fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
                    total += 1
                    url_stat[u][1] += 1
                else:
                    url_stat[u][0] += 1

            if 'passage' not in ret:
                skip_count += 1
                continue

    print('total err_json %s lines' % err_json)
    print("total skip %s lines" % skip_count)
    print("remain %s lines "  % total)
    for u in url_stat:
        url_stat[u][2] = 1.0 * url_stat[u][1] / (url_stat[u][1]+url_stat[u][0])
    url_stat_list = sorted(url_stat.items(), key = lambda x: x[1][2], reverse=True)
    fout = open('url_stat_list', 'w')
    for u in url_stat_list:
        if u[1][0]+u[1][1] > 100:
            fout.write(str(u)+'\n')
    fout.close()

def process_valid(filename, outfile):
    fo = codecs.open(outfile, 'w', 'utf8')
    skip_count = 0
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
            ret['qid'] = str(data['query_id'])
            ret['query'] = drop_punctuation(strQ2B(data['query']))
            ret['answers'] = []
            ps = data['passages']
            ans = drop_punctuation(strQ2B(data['answer'])).replace(' ', '')
            allp = []
            for p in ps:
                p = drop_punctuation(strQ2B(p['passage_text']))
                allp.append(p)
                if ans in p:
                    ans1 = {'answer_start':p.find(ans), 'text':ans}
                    ret['answers'].append(ans1)
                    ret['passage'] = p
                    break
            if 'passage' not in ret:
                #print(ret['query'])
                #print (ans)
                #print ('\n'.join(allp))
                #print("error data, skip")
                skip_count += 1
                continue
            total += 1
            fo.write(json.dumps(ret, ensure_ascii=False) + '\n')
    print('total err_json %s lines' % err_json)
    print("total skip %s lines" % skip_count)
    print("remain %s lines "  % total)

if __name__ == "__main__":
    if sys.argv[3] == "train":
        process_train(sys.argv[1], sys.argv[2])
    elif sys.argv[3] == "valid":
        process_valid(sys.argv[1], sys.argv[2])
