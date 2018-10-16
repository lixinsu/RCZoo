#!/bin/bash


set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH
MODELTYPE=${1}
MODEL=${MODELTYPE}_bl
DATASET=test-baseline
DIR=baseline

python3 scripts/reader/predict.py ${MODELTYPE} data/${DIR}/${DATASET}.json --model data/models/${MODEL}.mdl --out-dir data/${DIR}/ --tokenizer spacy --batch-size 64
python3 new_eval.py data/${DIR}/${DATASET}-${MODEL}.preds data/${DIR}/test.json
