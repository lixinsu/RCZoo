#!/bin/bash
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

mkdir data
mkdir data/embeddings
mkdir data/datasets
mkdir data/models

# Configure download location
DOWNLOAD_PATH="./data"

# Get externally hosted data
DATASET_PATH="$DOWNLOAD_PATH/datasets"

# Get SQuAD train
wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
python scripts/convert/squad.py "$DATASET_PATH/SQuAD-v1.1-train.json" "$DATASET_PATH/SQuAD-v1.1-train.txt"

# Get SQuAD dev
wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
python scripts/convert/squad.py "$DATASET_PATH/SQuAD-v1.1-dev.json" "$DATASET_PATH/SQuAD-v1.1-dev.txt"

