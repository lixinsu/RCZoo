#!/bin/bash
set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH
#python3 scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-train --tokenizer spacy
#python3 scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-dev --tokenizer spacy
python3 scripts/reader/train_multi_fusionnet.py --data-dir data/multispan --restrict-vocab 0 --train-file train-processed-spacy.txt --dev-file dev-processed-spacy.txt --dev-json dev.json --model-dir data/models --model-name multi_fusionnet --exp-id test2


