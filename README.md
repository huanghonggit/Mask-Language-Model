# BERT_mlm_pytorch


> BERT 2018 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## Introduction

Google AI's BERT paper shows the amazing result on various NLP task (new 17 NLP tasks SOTA), 
This paper proved that Transformer(self-attention) based encoder can be powerfully used as 
alternative of previous language model with proper language model training method. 
This repo is implementation of Mask LM in BERT. Code is very simple and easy to understand fastly.
Some of these codes are based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)


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



## Quick tour
### 0. Prepare your corpus
Some basic data processing, removing sentences that are not composed of English, punctuation, and numbers; and only retaining sentences with a character length between 30-512.
```
python preprocess.py -i data/corpus.txt -o data/corpus_pre.txt
```
```
Welcome to China\n
I can stay here all night\n
```

### 1. Building vocab based on your corpus
Here vocab is the common character we specify(Vocab(88)=letter(52)+punctuation(21)+logits(10)+pad+eos+sos+mask+unk
); But maybe vocab uses ASCII code 0-255 plus 5 special characters would be better.

```
python vocab.py -o data/vocab.test
```

### 2. Train your own MLM model
After preparing the training set ,test set and vocab, you could start training.
```
python __main__.py -c ./data/test_1w.txt -t ./data/test_1.txt -v ./data/vocab.test -o ./output
```

### 3. Valid your pretrain model
Ready for your pre-trained models and vocab, you can use only a few sentences to verify the convergence effect of the model.
```
python valid_model.py -m ./data/ep_0_mlm -v ./data/vocab.test
```
And you can also use the accuracy of downstream tasks such as classification tasks to determine the degree of convergence.
(https://github.com/huanghonggit/Finetune_MLM)

