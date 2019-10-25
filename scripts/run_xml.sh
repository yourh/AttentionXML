#!/usr/bin/env bash

python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 0
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 1
python main.py --data-cnf configure/datasets/$1.yaml --model-cnf configure/models/$2-$1.yaml -t 2
python ensemble.py -p results/$2-$1 -t 3
