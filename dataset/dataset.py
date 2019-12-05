from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np

def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')

def collate_mlm(batch):

    input_lens = [len(x[0]) for x in batch]
    max_x_len = max(input_lens)

    # chars
    chars_pad = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars_pad)

    # labels
    labels_pad = [pad1d(x[1], max_x_len) for x in batch]
    labels = np.stack(labels_pad)

    # position
    position = [pad1d(range(1, len + 1), max_x_len) for len in input_lens]
    position = np.stack(position)

    chars = torch.tensor(chars).long()
    labels = torch.tensor(labels).long()
    position = torch.tensor(position).long()

    output = {"mlm_input": chars,
              "mlm_label": labels,
              "input_position": position}

    return output

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = []
                for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.lines.append(line)
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        t = self.lines[item]
        t1_random, t1_label = self.random_word(t)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        mlm_input = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index] #  3，1，2
        mlm_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]

        return mlm_input, mlm_label


    def random_word(self, sentence):
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
                    chars[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    chars[i] = random.randrange(self.vocab.vocab_size)

                # 10% randomly change token to current token
                else:
                    chars[i] = self.vocab.char2index(char)

                output_label.append(self.vocab.char2index(char))

            else:
                chars[i] = self.vocab.char2index(char)
                output_label.append(0)

        return chars, output_label



    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
