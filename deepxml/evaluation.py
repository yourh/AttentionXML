#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import numpy as np
from functools import partial
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Union, Optional, List, Iterable, Hashable


__all__ = ['get_precision', 'get_p_1', 'get_p_3', 'get_p_5', 'get_p_10',
           'get_ndcg', 'get_n_1', 'get_n_3', 'get_n_5', 'get_n_10',
           'get_inv_propensity', 'get_psp',
           'get_psp_1', 'get_psp_3', 'get_psp_5', 'get_psp_10',
           'get_psndcg_1', 'get_psndcg_3', 'get_psndcg_5', 'get_psndcg_10']

TPredict = np.ndarray
TTarget = Union[Iterable[Iterable[Hashable]], csr_matrix]
TMlb = Optional[MultiLabelBinarizer]
TClass = Optional[List[Hashable]]


def get_mlb(classes: TClass = None, mlb: TMlb = None, targets: TTarget = None):
    if classes is not None:
        mlb = MultiLabelBinarizer(classes, sparse_output=True)
    if mlb is None and targets is not None:
        if isinstance(targets, csr_matrix):
            mlb = MultiLabelBinarizer(range(targets.shape[1]), sparse_output=True)
            mlb.fit(None)
        else:
            mlb = MultiLabelBinarizer(sparse_output=True)
            mlb.fit(targets)
    return mlb


def get_precision(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb, targets)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top])
    return prediction.multiply(targets).sum() / (top * targets.shape[0])


get_p_1 = partial(get_precision, top=1)
get_p_3 = partial(get_precision, top=3)
get_p_5 = partial(get_precision, top=5)
get_p_10 = partial(get_precision, top=10)


def get_ndcg(prediction: TPredict, targets: TTarget, mlb: TMlb = None, classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb, targets)
    log = 1.0 / np.log2(np.arange(top) + 2)
    dcg = np.zeros((targets.shape[0], 1))
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i: i+1])
        dcg += p.multiply(targets).sum(axis=-1) * log[i]
    return np.average(dcg / log.cumsum()[np.minimum(targets.sum(axis=-1), top) - 1])


get_n_1 = partial(get_ndcg, top=1)
get_n_3 = partial(get_ndcg, top=3)
get_n_5 = partial(get_ndcg, top=5)
get_n_10 = partial(get_ndcg, top=10)


def get_inv_propensity(train_y: csr_matrix, a=0.55, b=1.5):
    n, number = train_y.shape[0], np.asarray(train_y.sum(axis=0)).squeeze()
    c = (np.log(n) - 1) * ((b + 1) ** a)
    return 1.0 + c * (number + b) ** (-a)


def get_psp(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
            classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb)
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    prediction = mlb.transform(prediction[:, :top]).multiply(inv_w)
    num = prediction.multiply(targets).sum()
    t, den = csr_matrix(targets.multiply(inv_w)), 0
    for i in range(t.shape[0]):
        den += np.sum(np.sort(t.getrow(i).data)[-top:])
    return num / den


get_psp_1 = partial(get_psp, top=1)
get_psp_3 = partial(get_psp, top=3)
get_psp_5 = partial(get_psp, top=5)
get_psp_10 = partial(get_psp, top=10)


def get_psndcg(prediction: TPredict, targets: TTarget, inv_w: np.ndarray, mlb: TMlb = None,
               classes: TClass = None, top=5):
    mlb = get_mlb(classes, mlb)
    log = 1.0 / np.log2(np.arange(top) + 2)
    psdcg = 0.0
    if not isinstance(targets, csr_matrix):
        targets = mlb.transform(targets)
    for i in range(top):
        p = mlb.transform(prediction[:, i: i+1]).multiply(inv_w)
        psdcg += p.multiply(targets).sum() * log[i]
    t, den = csr_matrix(targets.multiply(inv_w)), 0.0
    for i in range(t.shape[0]):
        num = min(top, len(t.getrow(i).data))
        den += -np.sum(np.sort(-t.getrow(i).data)[:num] * log[:num])
    return psdcg / den


get_psndcg_1 = partial(get_psndcg, top=1)
get_psndcg_3 = partial(get_psndcg, top=3)
get_psndcg_5 = partial(get_psndcg, top=5)
get_psndcg_10 = partial(get_psndcg, top=10)
