#!/bin/bash

set -ex
echo `pwd`
export PYTHONPATH=`pwd`:$PYTHONPATH
MODEL=docqa_multi_debug
DATASET=dev
#python3 scripts/reader/predict_multi.py data/multispan/${DATASET}.json --model data/models/docqa_multi_debug.mdl --out-dir data/multispan/ --tokenizer spacy --batch-size 64
python3 new_eval.py data/multispan/${DATASET}-${MODEL}.preds data/multispan/${DATASET}.json
