#!/bin/bash

set -ex
echo `pwd`
export PYTHONPATH=`pwd`:$PYTHONPATH
MODEL=multi_fusionnet
DATASET=dev
python3 scripts/reader/predict_multi_fusionnet.py data/multispan/${DATASET}.json --model data/models/${MODEL}.mdl --out-dir data/multispan/ --tokenizer spacy --batch-size 64
python3 new_eval.py data/multispan/${DATASET}-${MODEL}.preds data/multispan/${DATASET}.json
