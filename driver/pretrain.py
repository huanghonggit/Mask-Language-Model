from module import optim4GPU
import seaborn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from model import BERTLM, BERT
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import config.hparams as hp
import sys
import traceback


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = hp.log_freq, args=None, global_step=0, path=None):
        """
        :param bert: MLM model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        self.args = args
        self.step = global_step
        self.path = path

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0,1,2,3" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        total_steps = hp.epochs * len(self.train_data)
        self.optimer = optim4GPU(self.model, total_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        # Writer
        self.log_freq = log_freq
        # train
        self.train_loss_writer = SummaryWriter(f'{self.path.runs_path}/train/train_loss')
        self.train_attn_layer_writer = SummaryWriter(f'{self.path.runs_path}/train/attn_layer')
        self.train_model_param_writer = SummaryWriter(f'{self.path.runs_path}/train/model_param')
        # valid
        self.valid_loss_writer = SummaryWriter(f'{self.path.runs_path}/valid/valid_loss')
        self.valid_attn_layer_writer = SummaryWriter(f'{self.path.runs_path}/valid/valid_attn_layer')

        self.num_params()

    def train(self):

        train_writer = (self.train_loss_writer, self.train_attn_layer_writer, self.train_model_param_writer)
        valid_writer = (self.valid_loss_writer, self.valid_attn_layer_writer)
        try:
            for epoch in range(hp.epochs):

                # Setting the tqdm progress bar
                data_iter = tqdm.tqdm(enumerate(self.train_data),
                                      desc="EP_%s:%d" % ("train", epoch),
                                      total=len(self.train_data),
                                      bar_format="{l_bar}{r_bar}")

                running_loss = 0
                for i, data in data_iter:

                    self.step += 1

                    # 0. batch_data will be sent into the device(GPU or cpu)
                    data = {key: value.to(self.device) for key, value in data.items()}

                    # 1. forward masked_lm model
                    mask_lm_output, attn_list = self.model.forward(data["mlm_input"], data["input_position"])

                    # 2. NLLLoss of predicting masked token word
                    self.optimer.zero_grad()
                    loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                    # 3. backward and optimization only in train
                    loss.backward()
                    self.optimer.step()

                    # loss
                    running_loss += loss.item()
                    avg_loss = running_loss / (i + 1)

                    # write log
                    post_fix = {
                        "epoch": epoch,
                        "iter": i,
                        "step": self.step,
                        "avg_loss": avg_loss,
                        "loss": loss.item()
                    }
                    if i % self.log_freq == 0:
                        data_iter.write(str(post_fix))

                    # writer train loss
                    if self.step % hp.save_train_loss == 0:
                        self.train_loss_writer.add_scalar('train_loss', loss, self.step)

                    # writer
                    if self.step % hp.save_runs == 0 and data["mlm_input"].size(0) == hp.batch_size: # 不足batch数量则不采样

                        # writer attns_layer
                        for layer, prob in enumerate(attn_list):
                            prob = prob[0]
                            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
                            print("Layer", layer + 1)
                            for h in range(hp.attn_heads):
                                # a = self.model.bert.layers[layer].multihead.attention[0][h].data
                                self.draw(prob[h].cpu().detach().numpy(),
                                     [], [], ax=axs[h])
                            plt.savefig(f"{self.path.plt_train_attn_path}/Epoch{epoch}_train_step{self.step}_layer{layer+1}")
                            # plt.show()

                        # tensorboardX write
                        for i, prob in enumerate(attn_list):  # 第i层,每层画四个图
                            prob = prob[0]
                            for j in range(hp.attn_heads):  # 1,2,3,4  第j个
                                # print(f"j * self.args.batch_size - 1:{j * self.args.batch_size - 1}")
                                x = vutils.make_grid(prob[j] * 255)  # eg:如果是512,94,94  则取127,255,383,511
                                self.train_attn_layer_writer.add_image(f'Epoch{epoch}_train_attn_layer{i}_head{j + 1}', x, self.step)

                        # write model_param
                        for name, param in self.model.module.named_parameters():  # param.clone().cpu().data.numpy()   .module
                            self.train_model_param_writer.add_histogram(f"Epoch{epoch}_train_{name}", param.clone().cpu().data.numpy(), self.step)

                    # save model checkpoint
                    if self.step % hp.save_checkpoint == 0:
                        self.bert.checkpoint(self.path.bert_checkpoints_path, self.step)

                    # save bert model
                    if self.step % hp.save_model == 0:
                        self.save_model(epoch, f"{self.path.bert_path}/bert")
                        self.save_mlm_model(epoch, f"{self.path.mlm_path}/mlm")

                    # evaluate
                    if self.step % hp.save_valid_loss == 0:
                        valid_loss = self.evaluate(epoch, valid_writer)


                valid_loss = self.evaluate(epoch, valid_writer)
                print(f"EP_{epoch}, train_avg_loss={avg_loss}, valid_avg_loss={valid_loss}")

            for writer in train_writer:
                writer.close()
            for writer in valid_writer:
                writer.close()

        except BaseException:
            traceback.print_exc()
            for writer in train_writer:
                writer.close()
            for writer in valid_writer:
                writer.close()



    def evaluate(self, epoch, valid_writer):
        (self.valid_loss_writer, self.valid_attn_layer_writer) = valid_writer
        self.model.eval()

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_%s:%d" % ("test", epoch),
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        running_loss = 0
        with torch.no_grad():
            for i, data in data_iter:

                self.step += 1

                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward masked_lm model
                mask_lm_output, attn_list = self.model.forward(data["mlm_input"], data["input_position"])

                # 2. NLLLoss of predicting masked token word
                loss = self.criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                # loss
                running_loss += loss.cpu().detach().numpy()
                avg_loss = running_loss / (i + 1)

                # print log
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "step": self.step,
                    "avg_loss": avg_loss,
                    "loss": loss.item()
                }
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))

                # writer valid loss
                self.valid_loss_writer.add_scalar('valid_loss', loss, self.step)

                if self.step % hp.save_runs == 0:
                    # writer attns_layer
                    for layer, prob in enumerate(attn_list):
                        prob = prob[0]
                        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
                        print("Layer", layer + 1)
                        for h in range(hp.attn_heads):
                            # a = self.model.bert.layers[layer].multihead.attention[0][h].data
                            self.draw(prob[h].cpu().detach().numpy(),
                                      [], [], ax=axs[h])
                        plt.savefig(
                            f"{self.path.plt_train_attn_path}/Epoch{epoch}_valid_step{self.step}_layer{layer + 1}")
                        # plt.show()

                    # tensorboardX write
                    for i, prob in enumerate(attn_list):  # 第i层,每层画四个图
                        prob = prob[0]
                        for j in range(hp.attn_heads):  # 1,2,3,4  第j个
                            # print(f"j * self.args.batch_size - 1:{j * self.args.batch_size - 1}")
                            x = vutils.make_grid(prob[j] * 255)  # eg:如果是512,94,94  则取127,255,383,511
                            self.train_attn_layer_writer.add_image(f'Epoch{epoch}_valid_attn_layer{i}_head{j + 1}',
                                                                   x, self.step)


            print(f"Valid Over!")
            return avg_loss


    def stream(self, message):
        sys.stdout.write(f"\r{message}")

    def draw(self, data, x, y, ax):
        seaborn.heatmap(data,
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,  # 取值0-1
                        cbar=False, ax=ax)

    def num_params(self, print_out=True):
        params_requires_grad = filter(lambda p: p.requires_grad, self.model.parameters())
        params_requires_grad = sum([np.prod(p.size()) for p in params_requires_grad]) #/ 1_000_000

        parameters = sum([np.prod(p.size()) for p in self.model.parameters()]) #/ 1_000_000
        if print_out:
            print('Trainable total Parameters: %d' % parameters)
            print('Trainable requires_grad Parameters: %d' % params_requires_grad)


    def save_model(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def save_mlm_model(self, epoch, file_path="output/mlm_trained.model"):
        """
        Saving the current MLM model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "_ep%d.model" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path








        # # write attn_head
        # for i, prob in enumerate(attn_list):  # 第i层 eg:512,100,100 没必要view成 128,4,100,100，直接取第0,1,2,3矩阵 # 100,100
        #     for j in range(4):  # 0,1,2,3   这里512的前第0,1,2,3个  100x100的矩阵  就是第一句话的4个head
        #         x = vutils.make_grid(prob[j] * 255)  # eg:如果是512,94,94  则取127,255,383,511
        #         self.writer4.add_image(f'Attention_enc_epoch{epoch}_layer{i}_head{j + 1}', x, self.step)

        # for i, prob in enumerate(attn_list):  # 第i层,每层画四个图
        #     for j in range(1, hp.attn_heads):  # 1,2,3,4  第j个
        #         # print(f"j * self.args.batch_size - 1:{j * self.args.batch_size - 1}")
        #         x = vutils.make_grid(prob[j * hp.batch_size - 1] * 255)  # eg:如果是512,94,94  则取127,255,383,511
        #         num = j * hp.batch_size - 1  # 画的是第几个位置的图
        #         self.train_attn_layer_writer.add_image(f'Attention_enc__epoch{epoch}_layer{i}_{num}', x, self.step)

        # # step
        # k = self.step // 1000
        #
        # # speed = i / (time.time() - start)
        #
        # # # save msg
        # # msg = f'| Epoch: {epoch}/{self.args.epochs} ({i}/{len(data_iter)}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
        # # self.stream(msg)


        # print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))




        # speed = i / (time.time() - start)
        #
        # # save msg
        # msg = f'| Epoch: {epoch}/{self.args.epochs} ({i}/{len(data_iter)}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
        # self.stream(msg)


        # for name, param in self.model.named_parameters(): # 参数包括了bert的参数
        #     print(name)

        # def adjust_learning_rate(self, optimizer, step_num, warmup_step=4000):
        #     # print(lr)
        #     lr_new = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
        #     print(lr_new)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_new


       # writer loss
            # if not train:
            #     self.writer.add_scalars('test_loss', {
            #         'loss': loss,
            #     }, i)



            # writer attention
            # for k, prob in enumerate(attn_list):
            #
            #     num_h = prob.size(0)  # b*h ;128  # 128*4=512   ; 96*4=384
            #     sample_rate = num_h // 24  # 4*h
            #     # sample_num = num_h // 32  # b：32
            #     for j in range(4):
            #         # print(f"第{j}个取样！")
            #         x = vutils.make_grid(prob[j * sample_rate] * 255)  #
            #         self.writer.add_image('Attention_%d_0' % i * epoch, x, k * 4 + j)

            # if i % 3 == 0:
            #     self.writer.add_scalars('train_loss', {
            #         'loss': loss,
            #     }, i)
            #
            #     # x = vutils.make_grid(attn_list[3][0] * 255)  # 取第6layer输出  atten里面 batch 最后一句
            #     # self.writer.add_image('Attention_%d_0' % i * epoch, x, i * epoch)
            #
            #     # writer log
            #     if i % self.log_freq == 0:
            #         data_iter.write(str(post_fix))

    # for layer in range(hp.attn_layers):
    #     fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    #     print("Layer", layer + 1)
    #     for h in range(hp.attn_heads):
    #         # a = self.model.bert.layers[layer].multihead.attention[0][h].data
    #         self.draw(self.model.module.bert.layers[layer].multihead.attention[0][h].data,  # [0, h].data,
    #                   [], [], ax=axs[h])
    #     plt.savefig(f"{self.path.plt_train_attn_path}/Epoch{epoch}_train_step{self.step}_layer{layer + 1}")
    #     plt.show()