#!/bin/bash
set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH
cd `pwd`
#python3 scripts/reader/preprocess.py data/multispan/ data/multispan/ --split train --tokenizer spacy
#python3 scripts/reader/preprocess.py data/multispan/ data/multispan/ --split dev --tokenizer spacy
python3 scripts/reader/train_multi_docqa.py --data-dir data/multispan --restrict-vocab 0 --train-file train-processed-spacy.txt --dev-file dev-processed-spacy.txt --dev-json dev.json --model-dir data/models --model-name docqa_multi_drop --exp-id test6


