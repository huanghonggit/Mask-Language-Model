import argparse
import sys
sys.path.extend(["../","./"])
import os
from torch.utils.data import DataLoader
from dataset import WordVocab
from model.bert import BERT
from dataset import BERTDataset,collate_mlm
from driver import BERTTrainer
from module import Paths
import torch
import numpy as np
import config.hparams as hp
import random


def train():
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--valid_dataset", required=True, type=str, help="valid set for evaluate train set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with vocab")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="ex)output/bert.model")

    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1, 2, 3], help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()
    set_seed(args)
    paths = Paths(args.output_path)

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", vocab.vocab_size)
    args.char_nums = vocab.vocab_size

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab,  corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Valid Dataset", args.valid_dataset)
    valid_dataset = BERTDataset(args.valid_dataset, vocab, on_memory=args.on_memory) \
        if args.valid_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=hp.batch_size, collate_fn=lambda batch: collate_mlm(batch),num_workers=args.num_workers, shuffle=False) # 训练语料按长度排好序的
    valid_data_loader = DataLoader(valid_dataset, batch_size=hp.batch_size, collate_fn=lambda batch: collate_mlm(batch), num_workers=args.num_workers, shuffle=False) \
        if valid_dataset is not None else None

    print("Building BERT model")
    bert = BERT(embed_dim=hp.embed_dim, hidden=hp.hidden, args=args)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, vocab.vocab_size, train_dataloader=train_data_loader, test_dataloader=valid_data_loader,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, args=args, path=paths)

    print("Training Start")

    trainer.train()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    train()






    # bert = BERT(vocab.vocab_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads) # len(vocab)

    # parser.add_argument("--clip_grad_norm", type=bool, default=True, help="clips the gradient norm: true, or false")
    # parser.add_argument("--mlm_clip_grad_norm", type=float, default=1.0, help="clips the gradient norm to prevent explosion - set to None if not needed")
    # parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of adam")
    # parser.add_argument("--adam_weight_decay", type=float, default=0, help="weight_decay of adam")
    # parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    # parser.add_argument("--adam_beta2", type=float, default=0.9, help="adam first beta value")


    # for epoch in range(args.epochs):
    #     trainer.train(epoch)
    #     if epoch % 3 == 0:
    #         trainer.save_model(epoch, f'{args.output_path}/mlm') # save model
    #
    #     if test_data_loader is not None:
    #         trainer.test(epoch)