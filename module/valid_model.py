import argparse

import seaborn
import torch
import matplotlib.pyplot as plt
import random
from dataset.vocab import WordVocab


def random_word(sentence, vocab):
    tokens = sentence.split()
    tokens_len = [len(token) for token in tokens]
    chars = [char for char in sentence]
    output_label = []

    for i, char in enumerate(chars):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                chars[i] = vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                chars[i] = random.randrange(vocab.vocab_size)

            # 10% randomly change token to current token
            else:
                chars[i] = vocab.char2index(char)

            output_label.append(vocab.char2index(char))

        else:
            chars[i] = vocab.char2index(char)
            output_label.append(0)

    return chars, output_label


def draw(data, x, y, ax):
    seaborn.heatmap(data,
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, # 取值0-1
                    cbar=False, ax=ax)


def Modelload(path):
    assert path is not None
    print(f"path:{path}")
    mlm_encoder = torch.load(path)
    return mlm_encoder


# 验证模型是否收敛
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", required=True, type=str, help="model of pretrain")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="path of vocab")
    args = parser.parse_args()

    model_path = args.model_path
    vocab_path = args.vocab_path

    vocab = WordVocab.load_vocab(vocab_path)

    model = torch.load(model_path)
    model.eval()

    sent = '_I _l _o _v _e _C _h _i _n _a _!'.split()

    text = 'I love China!'
    sent1, label = random_word(text, vocab)
    sent1 = torch.tensor(sent1).long().unsqueeze(0)
    mask_lm_output, attn_list = model.forward(sent1)

    chars = []
    for char in sent:
        chars.append(vocab.char2index(char))

    for layer in range(3):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Layer", layer+1)
        for h in range(4):
            # a = model.bert.layers[layer].multihead.attention[0,h].data
            draw(model.bert.layers[layer].multihead.attention[0, h].data, #[0, h].data,
                 sent, sent if h == 0 else [], ax=axs[h])
        plt.show()


if __name__ == '__main__':
    main()