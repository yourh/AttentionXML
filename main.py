#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import os
import click
import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from logzero import logger

from deepxml.dataset import MultiLabelDataset
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.models import Model
from deepxml.tree import FastAttentionXML
from deepxml.networks import AttentionRNN


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('-t', '--tree-id', type=click.INT, default=None)
def main(data_cnf, model_cnf, mode, tree_id):
    tree_id = F'-Tree-{tree_id}' if tree_id is not None else ''
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name, data_name = None, model_cnf['name'], data_cnf['name']
    model_path = os.path.join(model_cnf['path'], F'{model_name}-{data_name}{tree_id}')
    emb_init = get_word_emb(data_cnf['embedding']['emb_init'])
    logger.info(F'Model Name: {model_name}')

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
        if 'size' in data_cnf['valid']:
            random_state = data_cnf['valid'].get('random_state', 1240)
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_x)}')
        logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        if 'cluster' not in model_cnf:
            train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                      model_cnf['train']['batch_size'], shuffle=True, num_workers=4)
            valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = Model(network=AttentionRNN, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                          **data_cnf['model'], **model_cnf['model'])
            model.train(train_loader, valid_loader, **model_cnf['train'])
        else:
            model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
            model.train(train_x, train_y, valid_x, valid_y, mlb)
        logger.info('Finish Training')

    if mode is None or mode == 'eval':
        logger.info('Loading Test Set')
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        logger.info('Predicting')
        if 'cluster' not in model_cnf:
            test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'],
                                     num_workers=4)
            if model is None:
                model = Model(network=AttentionRNN, labels_num=labels_num, model_path=model_path, emb_init=emb_init,
                              **data_cnf['model'], **model_cnf['model'])
            scores, labels = model.predict(test_loader, k=model_cnf['predict'].get('k', 100))
        else:
            if model is None:
                model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
            scores, labels = model.predict(test_x)
        logger.info('Finish Predicting')
        labels = mlb.classes_[labels]
        output_res(data_cnf['output']['res'], F'{model_name}-{data_name}{tree_id}', scores, labels)


if __name__ == '__main__':
    main()
