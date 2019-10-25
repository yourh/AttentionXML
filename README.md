# AttentionXML
[AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727)

## Requirements

* python==3.7.4
* click==7.0
* ruamel.yaml==0.16.5
* numpy==1.16.2
* scipy==1.3.1
* scikit-learn==0.21.2
* gensim==3.4.0
* torch==1.0.1
* nltk==3.4
* tqdm==4.31.1
* joblib==0.13.2
* logzero==1.5.0

## Datasets

* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
* [Amazon-3M](https://drive.google.com/open?id=187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4)

Download the GloVe embedding (840B,300d) and convert it to gensim format (which can be loaded by **gensim.models.KeyedVectors.load**).

We also provide a converted GloVe embedding at [here](https://drive.google.com/file/d/10w_HuLklGc8GA_FtUSdnHT8Yo1mxYziP/view?usp=sharing). 

## XML Experiments

XML experiments in paper can be run directly such as:
```bash
./scripts/run_eurlex.sh
```
## Preprocess

Run preprocess.py for train and test datasets with tokenized texts as follows:
```bash
python preprocess.py \
--text-path data/EUR-Lex/train_texts.txt \
--label-path data/EUR-Lex/train_labels.txt \
--vocab-path data/EUR-Lex/vocab.npy \
--emb-path data/EUR-Lex/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

python preprocess.py \
--text-path data/EUR-Lex/test_texts.txt \
--label-path data/EUR-Lex/test_labels.txt \
--vocab-path data/EUR-Lex/vocab.npy 
```

Or run preprocss.py including tokenizing the raw texts by NLTK as follows:
```bash
python preprocess.py \
--text-path data/Wiki10-31K/train_raw_texts.txt \
--tokenized-path data/Wiki10-31K/train_texts.txt \
--label-path data/Wiki10-31K/train_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy \
--emb-path data/Wiki10-31K/emb_init.npy \
--w2v-model data/glove.840B.300d.gensim

python preprocess.py \
--text-path data/Wiki10-31K/test_raw_texts.txt \
--tokenized-path data/Wiki10-31K/test_texts.txt \
--label-path data/Wiki10-31K/test_labels.txt \
--vocab-path data/Wiki10-31K/vocab.npy 
```


## Train and Predict

Train and predict as follows:
```bash
python main.py --data-cnf configure/datasets/EUR-Lex.yaml --model-cnf configure/models/AttentionXML-EUR-Lex.yaml 
```

Or do prediction only with option "--mode eval".

## Ensemble

Train and predict with an ensemble:
```bash
python main.py --data-cnf configure/datasets/Wiki-500K.yaml --model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 0
python main.py --data-cnf configure/datasets/Wiki-500K.yaml --model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 1
python main.py --data-cnf configure/datasets/Wiki-500K.yaml --model-cnf configure/models/FastAttentionXML-Wiki-500K.yaml -t 2
python ensemble.py -p results/FastAttentionXML-Wiki-500K -t 3
```

## Evaluation

```bash
python evaluation.py --results results/AttentionXML-EUR-Lex-labels.npy --targets data/EUR-Lex/test_labels.npy
```
Or get propensity scored metrics together:

```bash
python evaluation.py \
--results results/FastAttentionXML-Amazon-670K-labels.npy \
--targets data/Amazon-670K/test_labels.npy \
--train-labels data/Amazon-670K/train_labels.npy \
-a 0.6 \
-b 2.6

```

## Reference
You et al., [AttentionXML: Label Tree-based Attention-Aware Deep Model for High-Performance Extreme Multi-Label Text Classification](https://arxiv.org/abs/1811.01727), NeurIPS 2019

## Declaration
It is free for non-commercial use. For commercial use, please contact Mr. Ronghi You and Prof. Shanfeng Zhu (zhusf@fudan.edu.cn).