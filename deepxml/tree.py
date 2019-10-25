#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2019/2/26
@author yrh

"""

import os
import time
import numpy as np
import torch
from multiprocessing import Process
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger

from deepxml.data_utils import get_word_emb
from deepxml.dataset import MultiLabelDataset, XMLDataset
from deepxml.models import Model, XMLModel
from deepxml.cluster import build_tree_by_level
from deepxml.networks import *


__all__ = ['FastAttentionXML']


class FastAttentionXML(object):
    """

    """
    def __init__(self, labels_num, data_cnf, model_cnf, tree_id=''):
        self.data_cnf, self.model_cnf = data_cnf.copy(), model_cnf.copy()
        model_name, data_name = model_cnf['name'], data_cnf['name']
        self.model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}')
        self.emb_init, self.level = get_word_emb(data_cnf['embedding']['emb_init']), model_cnf['level']
        self.labels_num, self.models = labels_num, {}
        self.inter_group_size, self.top = model_cnf['k'], model_cnf['top']
        self.groups_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}-cluster')

    @staticmethod
    def get_mapping_y(groups, labels_num, *args):
        mapping = np.empty(labels_num + 1, dtype=np.long)
        for idx, labels_list in enumerate(groups):
            mapping[labels_list] = idx
        mapping[labels_num] = len(groups)
        return (FastAttentionXML.get_group_y(mapping, y, len(groups)) for y in args)

    @staticmethod
    def get_group_y(mapping: np.ndarray, data_y: csr_matrix, groups_num):
        r, c, d = [], [], []
        for i in range(data_y.shape[0]):
            g = np.unique(mapping[data_y.indices[data_y.indptr[i]: data_y.indptr[i + 1]]])
            r += [i] * len(g)
            c += g.tolist()
            d += [1] * len(g)
        return csr_matrix((d, (r, c)), shape=(data_y.shape[0], groups_num))

    def train_level(self, level, train_x, train_y, valid_x, valid_y):
        model_cnf, data_cnf = self.model_cnf, self.data_cnf
        if level == 0:
            while not os.path.exists(F'{self.groups_path}-Level-{level}.npy'):
                time.sleep(30)
            groups = np.load(F'{self.groups_path}-Level-{level}.npy')
            train_y, valid_y = self.get_mapping_y(groups, self.labels_num, train_y, valid_y)
            labels_num = len(groups)
            train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                      model_cnf['train'][level]['batch_size'], num_workers=4, shuffle=True)
            valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = Model(AttentionRNN, labels_num=labels_num, model_path=F'{self.model_path}-Level-{level}',
                          emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            if not os.path.exists(model.model_path):
                logger.info(F'Training Level-{level}, Number of Labels: {labels_num}')
                model.train(train_loader, valid_loader, **model_cnf['train'][level])
                model.optimizer = None
                logger.info(F'Finish Training Level-{level}')
            self.models[level] = model
            logger.info(F'Generating Candidates for Level-{level+1}, '
                        F'Number of Labels: {labels_num}, Top: {self.top}')
            train_loader = DataLoader(MultiLabelDataset(train_x), model_cnf['valid']['batch_size'], num_workers=4)
            return train_y, model.predict(train_loader, k=self.top), model.predict(valid_loader, k=self.top)
        else:
            train_group_y, train_group, valid_group = self.train_level(level - 1, train_x, train_y, valid_x, valid_y)
            torch.cuda.empty_cache()

            logger.info('Getting Candidates')
            _, group_labels = train_group
            group_candidates = np.empty((len(train_x), self.top), dtype=np.int)
            for i, labels in tqdm(enumerate(group_labels), leave=False, desc='Parents'):
                ys, ye = train_group_y.indptr[i], train_group_y.indptr[i + 1]
                positive = set(train_group_y.indices[ys: ye])
                if self.top >= len(positive):
                    candidates = positive
                    for la in labels:
                        if len(candidates) == self.top:
                            break
                        if la not in candidates:
                            candidates.add(la)
                else:
                    candidates = set()
                    for la in labels:
                        if la in positive:
                            candidates.add(la)
                        if len(candidates) == self.top:
                            break
                    if len(candidates) < self.top:
                        candidates = (list(candidates) + list(positive - candidates))[:self.top]
                group_candidates[i] = np.asarray(list(candidates))

            if level < self.level - 1:
                while not os.path.exists(F'{self.groups_path}-Level-{level}.npy'):
                    time.sleep(30)
                groups = np.load(F'{self.groups_path}-Level-{level}.npy')
                train_y, valid_y = self.get_mapping_y(groups, self.labels_num, train_y, valid_y)
                labels_num, last_groups = len(groups), self.get_inter_groups(len(groups))
            else:
                groups, labels_num = None, train_y.shape[1]
                last_groups = np.load(F'{self.groups_path}-Level-{level-1}.npy')

            train_loader = DataLoader(XMLDataset(train_x, train_y, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_candidates),
                                      model_cnf['train'][level]['batch_size'], num_workers=4, shuffle=True)
            group_scores, group_labels = valid_group
            valid_loader = DataLoader(XMLDataset(valid_x, valid_y, training=False, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_labels,
                                                 group_scores=group_scores),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = XMLModel(network=FastAttentionRNN, labels_num=labels_num, emb_init=self.emb_init,
                             model_path=F'{self.model_path}-Level-{level}', **data_cnf['model'], **model_cnf['model'])
            if not os.path.exists(model.model_path):
                logger.info(F'Loading parameters of Level-{level} from Level-{level-1}')
                last_model = self.get_last_models(level - 1)
                model.network.module.emb.load_state_dict(last_model.module.emb.state_dict())
                model.network.module.lstm.load_state_dict(last_model.module.lstm.state_dict())
                model.network.module.linear.load_state_dict(last_model.module.linear.state_dict())
                logger.info(F'Training Level-{level}, '
                            F'Number of Labels: {labels_num}, '
                            F'Candidates Number: {train_loader.dataset.candidates_num}')
                model.train(train_loader, valid_loader, **model_cnf['train'][level])
                model.optimizer = model.state = None
                logger.info(F'Finish Training Level-{level}')
            self.models[level] = model
            if level == self.level - 1:
                return
            logger.info(F'Generating Candidates for Level-{level+1}, '
                        F'Number of Labels: {labels_num}, Top: {self.top}')
            group_scores, group_labels = train_group
            train_loader = DataLoader(XMLDataset(train_x, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_labels,
                                                 group_scores=group_scores),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            return train_y, model.predict(train_loader, k=self.top), model.predict(valid_loader, k=self.top)

    def get_last_models(self, level):
        return self.models[level].model if level == 0 else self.models[level].network

    def predict_level(self, level, test_x, k, labels_num):
        data_cnf, model_cnf = self.data_cnf, self.model_cnf
        model = self.models.get(level, None)
        if level == 0:
            logger.info(F'Predicting Level-{level}, Top: {k}')
            if model is None:
                model = Model(AttentionRNN, labels_num=labels_num, model_path=F'{self.model_path}-Level-{level}',
                              emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'],
                                     num_workers=4)
            return model.predict(test_loader, k=k)
        else:
            if level == self.level - 1:
                groups = np.load(F'{self.groups_path}-Level-{level-1}.npy')
            else:
                groups = self.get_inter_groups(labels_num)
            group_scores, group_labels = self.predict_level(level - 1, test_x, self.top, len(groups))
            torch.cuda.empty_cache()
            logger.info(F'Predicting Level-{level}, Top: {k}')
            if model is None:
                model = XMLModel(network=FastAttentionRNN, labels_num=labels_num,
                                 model_path=F'{self.model_path}-Level-{level}',
                                 emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            test_loader = DataLoader(XMLDataset(test_x, labels_num=labels_num,
                                                groups=groups, group_labels=group_labels, group_scores=group_scores),
                                     model_cnf['predict']['batch_size'], num_workers=4)
            return model.predict(test_loader, k=k)

    def get_inter_groups(self, labels_num):
        assert labels_num % self.inter_group_size == 0
        return np.asarray([list(range(i, i + self.inter_group_size))
                           for i in range(0, labels_num, self.inter_group_size)])

    def train(self, train_x, train_y, valid_x, valid_y, mlb):
        self.model_cnf['cluster']['groups_path'] = self.groups_path
        cluster_process = Process(target=build_tree_by_level,
                                  args=(self.data_cnf['train']['sparse'], self.data_cnf['train']['labels'], mlb),
                                  kwargs=self.model_cnf['cluster'])
        cluster_process.start()
        self.train_level(self.level - 1, train_x, train_y, valid_x, valid_y)
        cluster_process.join()
        cluster_process.close()

    def predict(self, test_x):
        return self.predict_level(self.level - 1, test_x, self.model_cnf['predict'].get('k', 100), self.labels_num)
