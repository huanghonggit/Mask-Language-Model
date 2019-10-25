# BERT_mlm_pytorch


> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## Introduction

Google AI's BERT paper shows the amazing result on various NLP task (new 17 NLP tasks SOTA), 
including outperform the human F1 score on SQuAD v1.1 QA task. 
This paper proved that Transformer(self-attention) based encoder can be powerfully used as 
alternative of previous language model with proper language model training method. 
And more importantly, they showed us that this pre-trained language model can be transfer 
into any NLP task without making task specific model architecture.

This repo is implementation of BERT. Code is very simple and easy to understand fastly.
Some of these codes are based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)



### 0. Prepare your corpus
```
Welcome to the \t the jungle\n
I can stay \t here all night\n
```


### 1. Building vocab based on your corpus
```shell
python vocab.py -c data/corpus.small -o data/vocab.small
```

### 2. Train your own MLM model
```shell
python main.py -c data/corpus.small -v data/vocab.small -o output/bert.model
```

## Language Model Pre-training

In the paper, authors shows the new language model training methods,which are "masked language model" and "predict next sentence".
Only "masked language model" is implemented here.


### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

