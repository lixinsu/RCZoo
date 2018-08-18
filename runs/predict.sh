#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
cd `pwd`
DATA=test
python3 scripts/reader/converter.py data/sogou/${DATA}.json data/sogou/${DATA}_std_all.json test
python3 scripts/reader/predict.py data/sogou/${DATA}_std_all.json --model data/models/docqa_sogou.mdl --tokenizer jieba --out-dir data/sogou
python3 scripts/reader/merge_result.py data/sogou/${DATA}_std_all-docqa_sogou.preds

