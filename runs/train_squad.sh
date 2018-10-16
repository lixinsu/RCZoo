#!/bin/bash
set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH

MODEL=$1

models="bidaf drqa slqa fusionnet docqa qanet"
valid_model=false
for m in $models
do
    echo $m
    if [ "$m" = "$MODEL" ]; then
        valid_model=true
    fi
done
if [ "$valid_model" = true ] ; then
    echo "train the model $MODEL"
else
    echo "The $MODEL model is not supported now"
    exit 1
fi

python3 scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-train --tokenizer spacy
python3 scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-dev --tokenizer spacy

python3 scripts/reader/train_${MODEL}.py --data-dir data/datasets \
                                        --restrict-vocab 0 --train-file SQuAD-v1.1-train-processed-spacy.txt \
                                        --dev-file SQuAD-v1.1-dev-processed-spacy.txt \
                                        --dev-json SQuAD-v1.1-dev.json \
                                        --model-dir data/models \
                                        --model-name ${MODEL}_squad_elmo_v1 \
                                        --num-epochs 20 \
                                        #--exp-id test


