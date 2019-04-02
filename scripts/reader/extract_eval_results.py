#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import numpy as np
import pandas as pd

def extract_file(logfile, max_epoch=40):
    with open(logfile) as infp:
        pat = re.compile(r"Epoch = ([0-9]+) \| EM = ([0-9]+\.[0-9]+) \| F1 = ([0-9]+\.[0-9]+)")
        res = []
        for line in infp:
            if "dev valid official" in line:
                m = pat.search(line)
                res.append([m.group(1),m.group(2),m.group(3)])
    return res[:max_epoch]


def compare_result(files):
    results = {}
    for ifile in files:
        print(ifile.split('/')[-1])
        save_name = ifile.split('/')[-1].split('.')[0]
        res = extract_file(ifile)
        results['%s-EM' % save_name] = [float(ires[1]) for ires in res]
        results['%s-F1' % save_name] = [float(ires[2]) for ires in res]
    pd.DataFrame.from_dict(results).to_csv('compare.csv', sep=',')


if __name__ == '__main__':
    compare_result(sys.argv[1:])
