#!/usr/bin/env python
# coding: utf-8
import os
import sys
import copy
import yaml
import random
import argparse
from subprocess import check_call

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/', required=True,
                    help='Directory containing params.yaml')
parser.add_argument('--random', action='store_true',
                    help='Random hyper-parameter search')
parser.add_argument('--ratio', type=float, default=0.5,
                    help='Random hyper-parameter search')


def launch_training_job(parent_dir, job_name, params):
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    json_path = os.path.join(model_dir, 'params.yaml')
    yaml.dump(params, open(json_path, 'w'), default_flow_style=False)
    cmd = "{python} scripts/reader/train_bidaf.py --model-dir {model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


def _format_name(ks, vs):
    return '_'.join(['{}-{}'.format(k,v) for k,v in zip(ks, vs)])


def dfs_params(params, param_keys, param_values):
    if len(param_keys) == len(param_values):
        job_name = _format_name(param_keys, param_values)
        run_params = copy.copy(basic_params)
        run_params.update(dict(zip(param_keys, param_values)))
        launch_training_job(args.parent_dir, job_name, run_params)
        return
    for param in params[param_keys[len(param_values)]]:
        param_values.append(param)
        dfs_params(params, param_keys, param_values)
        param_values.pop()


def _count_cases(params):
    rv = 1
    for v in params.values():
            rv *= len(v)
    return rv


def random_params(params, param_keys, ratio=0.5):
    total = _count_cases(params)
    total =  int(total * ratio)
    tried = set()
    while len(tried) < total:
        param_values = []
        for key in param_keys:
            param_values.append(random.choice(params[key]))
        if tuple(param_values) in tried:
            continue
        tried.add(tuple(param_values))
        job_name  = _format_name(param_keys, param_values)
        run_params = copy.copy(basic_params)
        run_params.update(dict(zip(param_keys, param_values)))
        launch_training_job(args.parent_dir, job_name, run_params)


if __name__ == "__main__":
    args = parser.parse_args()
    param_path = os.path.join(args.parent_dir, 'params.yaml')
    assert os.path.isfile(param_path), "No yaml configuration file found at {}".format(params_path)
    basic_params = yaml.load(open(param_path))
    tuned_param_path = os.path.join(args.parent_dir, 'tuned_params.yaml')
    assert os.path.isfile(tuned_param_path), "No tuned params configuration file"
    tuned_params = yaml.load(open(tuned_param_path))
    param_keys = list(tuned_params.keys())
    if not args.random:
        dfs_params(tuned_params, param_keys, [])
    else:
        random_params(tuned_params, param_keys, ratio=args.ratio)
