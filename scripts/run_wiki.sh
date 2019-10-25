#!/usr/bin/env bash

DATA=Wiki-500K
MODEL=FastAttentionXML

./scripts/run_preprocess.sh $DATA
./scripts/run_xml.sh $DATA $MODEL

python evaluation.py \
--results results/$MODEL-$DATA-Ensemble-labels.npy \
--targets data/$DATA/test_labels.npy \
--train-labels data/$DATA/train_labels.npy \
-a 0.5 \
-b 0.4
