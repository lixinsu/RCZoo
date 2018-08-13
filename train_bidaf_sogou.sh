#!/bin/bash
set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH
#python3 scripts/reader/converter.py data/sogou/train.json data/sogou/train_std.json  train
#python3 scripts/reader/converter.py data/sogou/dev.json data/sogou/dev_std.json  train
#
#python3 scripts/reader/preprocess.py data/sogou/ data/sogou/ --split train_std --tokenizer jieba
#python3 scripts/reader/preprocess.py data/sogou/ data/sogou/ --split dev_std --tokenizer jieba
#
python3 scripts/reader/train_bidaf.py --num-epochs 40 --embedding-file  sogou.full.embed  --data-dir data/sogou --restrict-vocab 0 --train-file  train_std-processed-jieba.txt --dev-file dev_std-processed-jieba.txt --dev-json dev_std.json --model-dir data/models --model-name bidaf_sogou 



