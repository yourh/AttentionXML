#!/usr/bin/env bash

DATA=AmazonCat-13K
MODEL=AttentionXML

./scripts/run_preprocess.sh $DATA
./scripts/run_xml.sh $DATA $MODEL

python evaluation.py \
--results results/$MODEL-$DATA-Ensemble-labels.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy
