
set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH

MODEL=$1

#python scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-train --tokenizer spacy
#python scripts/reader/preprocess.py data/datasets/ data/datasets/ --split SQuAD-v1.1-dev --tokenizer spacy

python scripts/reader/train_${MODEL}.py --data-dir data/datasets \
                                        --restrict-vocab 0 --train-file SQuAD-v1.1-train-processed-spacy.txt \
                                        --dev-file SQuAD-v1.1-dev-processed-spacy.txt \
                                        --dev-json SQuAD-v1.1-dev.json \
                                        --model-dir data/models/${MODEL} \
                                        --model-name test2 \
                                        --num-epochs 20 \
                                        --batch-size 32
                                        #--exp-id test


