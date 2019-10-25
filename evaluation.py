#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2019/8/21
@author yrh

"""

import warnings
warnings.filterwarnings('ignore')

import click
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from deepxml.evaluation import *


@click.command()
@click.option('-r', '--results', type=click.Path(exists=True), help='Path of results.')
@click.option('-t', '--targets', type=click.Path(exists=True), help='Path of targets.')
@click.option('--train-labels', type=click.Path(exists=True), default=None, help='Path of labels for training set.')
@click.option('-a', type=click.FLOAT, default=0.55, help='Parameter A for propensity score.')
@click.option('-b', type=click.FLOAT, default=1.5, help='Parameter B for propensity score.')
def main(results, targets, train_labels, a, b):
    res, targets = np.load(results), np.load(targets)
    mlb = MultiLabelBinarizer(sparse_output=True)
    targets = mlb.fit_transform(targets)
    print('Precision@1,3,5:', get_p_1(res, targets, mlb), get_p_3(res, targets, mlb), get_p_5(res, targets, mlb))
    print('nDCG@1,3,5:', get_n_1(res, targets, mlb), get_n_3(res, targets, mlb), get_n_5(res, targets, mlb))
    if train_labels is not None:
        train_labels = np.load(train_labels)
        inv_w = get_inv_propensity(mlb.transform(train_labels), a, b)
        print('PSPrecision@1,3,5:', get_psp_1(res, targets, inv_w, mlb), get_psp_3(res, targets, inv_w, mlb),
              get_psp_5(res, targets, inv_w, mlb))
        print('PSnDCG@1,3,5:', get_psndcg_1(res, targets, inv_w, mlb), get_psndcg_3(res, targets, inv_w, mlb),
              get_psndcg_5(res, targets, inv_w, mlb))


if __name__ == '__main__':
    main()
