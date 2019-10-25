#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2018/12/9
@author yrh

"""

import os
import numpy as np
import joblib
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer, normalize
from sklearn.datasets import load_svmlight_file
from gensim.models import KeyedVectors
from tqdm import tqdm
from typing import Union, Iterable


__all__ = ['build_vocab', 'get_data', 'convert_to_binary', 'truncate_text', 'get_word_emb', 'get_mlb',
           'get_sparse_feature', 'output_res']


def build_vocab(texts: Iterable, w2v_model: Union[KeyedVectors, str], vocab_size=500000,
                pad='<PAD>', unknown='<UNK>', sep='/SEP/', max_times=1, freq_times=1):
    if isinstance(w2v_model, str):
        w2v_model = KeyedVectors.load(w2v_model)
    emb_size = w2v_model.vector_size
    vocab, emb_init = [pad, unknown], [np.zeros(emb_size), np.random.uniform(-1.0, 1.0, emb_size)]
    counter = Counter(token for t in texts for token in set(t.split()))
    for word, freq in sorted(counter.items(), key=lambda x: (x[1], x[0] in w2v_model), reverse=True):
        if word in w2v_model or freq >= freq_times:
            vocab.append(word)
            # We used embedding of '.' as embedding of '/SEP/' symbol.
            word = '.' if word == sep else word
            emb_init.append(w2v_model[word] if word in w2v_model else np.random.uniform(-1.0, 1.0, emb_size))
        if freq < max_times or vocab_size == len(vocab):
            break
    return np.asarray(vocab), np.asarray(emb_init)


def get_word_emb(vec_path, vocab_path=None):
    if vocab_path is not None:
        with open(vocab_path) as fp:
            vocab = {word: idx for idx, word in enumerate(fp)}
        return np.load(vec_path), vocab
    else:
        return np.load(vec_path)


def get_data(text_file, label_file=None):
    return np.load(text_file), np.load(label_file) if label_file is not None else None


def convert_to_binary(text_file, label_file=None, max_len=None, vocab=None, pad='<PAD>', unknown='<UNK>'):
    with open(text_file) as fp:
        texts = np.asarray([[vocab.get(word, vocab[unknown]) for word in line.split()]
                           for line in tqdm(fp, desc='Converting token to id', leave=False)])
    labels = None
    if label_file is not None:
        with open(label_file) as fp:
            labels = np.asarray([[label for label in line.split()]
                                 for line in tqdm(fp, desc='Converting labels', leave=False)])
    return truncate_text(texts, max_len, vocab[pad], vocab[unknown]), labels


def truncate_text(texts, max_len=500, padding_idx=0, unknown_idx=1):
    if max_len is None:
        return texts
    texts = np.asarray([list(x[:max_len]) + [padding_idx] * (max_len - len(x)) for x in texts])
    texts[(texts == padding_idx).all(axis=1), 0] = unknown_idx
    return texts


def get_mlb(mlb_path, labels=None) -> MultiLabelBinarizer:
    if os.path.exists(mlb_path):
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def get_sparse_feature(feature_file, label_file):
    sparse_x, _ = load_svmlight_file(feature_file, multilabel=True)
    return normalize(sparse_x), np.load(label_file) if label_file is not None else None


def output_res(output_path, name, scores, labels):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, F'{name}-scores'), scores)
    np.save(os.path.join(output_path, F'{name}-labels'), labels)
