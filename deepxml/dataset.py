#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/10
@author yrh

"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Sequence, Optional, Union


__all__ = ['MultiLabelDataset', 'XMLDataset']

TDataX = Sequence[Sequence]
TDataY = Optional[csr_matrix]
TCandidate = TGroup = Optional[np.ndarray]
TGroupLabel = TGroupScore = Optional[Union[np.ndarray, torch.Tensor]]


class MultiLabelDataset(Dataset):
    """

    """
    def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True):
        self.data_x, self.data_y, self.training = data_x, data_y, training

    def __getitem__(self, item):
        data_x = self.data_x[item]
        if self.training and self.data_y is not None:
            data_y = self.data_y[item].toarray().squeeze(0).astype(np.float32)
            return data_x, data_y
        else:
            return data_x

    def __len__(self):
        return len(self.data_x)


class XMLDataset(MultiLabelDataset):
    """

    """
    def __init__(self, data_x: TDataX, data_y: TDataY = None, training=True,
                 labels_num=None, candidates: TCandidate = None, candidates_num=None,
                 groups: TGroup = None, group_labels: TGroupLabel = None, group_scores: TGroupScore = None):
        super(XMLDataset, self).__init__(data_x, data_y, training)
        self.labels_num, self.candidates, self.candidates_num = labels_num, candidates, candidates_num
        self.groups, self.group_labels, self.group_scores = groups, group_labels, group_scores
        if self.candidates is None:
            self.candidates = [np.concatenate([self.groups[g] for g in group_labels])
                               for group_labels in tqdm(self.group_labels, leave=False, desc='Candidates')]
            if self.group_scores is not None:
                self.candidates_scores = [np.concatenate([[s] * len(self.groups[g])
                                                          for g, s in zip(group_labels, group_scores)])
                                          for group_labels, group_scores in zip(self.group_labels, self.group_scores)]
        else:
            self.candidates_scores = [np.ones_like(candidates) for candidates in self.candidates]
        if self.candidates_num is None:
            self.candidates_num = self.group_labels.shape[1] * max(len(g) for g in groups)

    def __getitem__(self, item):
        data_x, candidates = self.data_x[item], np.asarray(self.candidates[item], dtype=np.int)
        if self.training and self.data_y is not None:
            if len(candidates) < self.candidates_num:
                sample = np.random.randint(self.labels_num, size=self.candidates_num - len(candidates))
                candidates = np.concatenate([candidates, sample])
            elif len(candidates) > self.candidates_num:
                candidates = np.random.choice(candidates, self.candidates_num, replace=False)
            data_y = self.data_y[item, candidates].toarray().squeeze(0).astype(np.float32)
            return (data_x, candidates), data_y
        else:
            scores = self.candidates_scores[item]
            if len(candidates) < self.candidates_num:
                scores = np.concatenate([scores, [-np.inf] * (self.candidates_num - len(candidates))])
                candidates = np.concatenate([candidates, [self.labels_num] * (self.candidates_num - len(candidates))])
            scores = np.asarray(scores, dtype=np.float32)
            return data_x, candidates, scores

    def __len__(self):
        return len(self.data_x)
