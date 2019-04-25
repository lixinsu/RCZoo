#!/usr/bin/env python
# coding: utf-8

"""Aggregates results from the preditions.json in a parent folder"""

import os
import sys
import argparse
import json
import re

from tabulate import tabulate
import subprocess
import shlex
from subprocess import run


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments',
                    help='Directory containing results of experiments')



def parse_result(filename):

    def _parse(line):
        pat = re.compile('EM\s=\s([0-9]+\.[0-9]+).+F1 = ([0-9]+\.[0-9]+)\s')
        m = pat.search(line)
        return m.group(1), m.group(2)

    lines =  open(filename).readlines()
    em, f1 = 0, 0
    for i, l in enumerate(lines):
        if 'Best valid' in l:
            em, f1 = _parse(lines[i-1])
    return em, f1


def aggregate_metrics(parent_dir, metrics):
    """Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/model_ckpt.txt`
    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    result_file = os.path.join(parent_dir, 'model_ckpt.txt')
    if os.path.isfile(result_file):
        em, f1 = parse_result(result_file)
        metrics[parent_dir] = {'EM': em, 'F1': f1}

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def metrics_to_table(metrics):
    # Get the headers from the first subdir. Assumes everything has the same metrics
    headers = metrics[list(metrics.keys())[0]].keys()
    table = [[subdir] + [values[h] for h in headers] for subdir, values in metrics.items()]
    res = tabulate(table, headers, tablefmt='pipe')
    return res


if __name__ == "__main__":
    args = parser.parse_args()

    # Aggregate metrics from args.parent_dir directory
    metrics = dict()
    aggregate_metrics(args.parent_dir, metrics)
    table = metrics_to_table(metrics)

    # Display the table to terminal
    print(table)

    # Save results in parent_dir/results.md
    save_file = os.path.join(args.parent_dir, "results.md")
    with open(save_file, 'w') as f:
        f.write(table)
