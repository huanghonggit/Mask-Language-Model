import torch.nn as nn
from model.bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__() # BERTLM
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)
        self.init_model()

    def forward(self, x, pos):
        x, attn_list = self.bert(x, pos)
        return self.mask_lm(x), attn_list

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)


        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    # def get_step(self):
    #     return self.step.data.item()
    #
    # def reset_step(self):
    #     self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
    #
    # def checkpoint(self, path):
    #     k_steps = self.get_step() // 1000
    #     self.save(f'{path}/checkpoint_{k_steps}k_steps.pyt')
    #
    # def log(self, path, msg):
    #     with open(path, 'a') as f:
    #         print(msg, file=f)
    #
    # def restore(self, path):
    #     if not os.path.exists(path):
    #         print('\nNew Tacotron Training Session...\n')
    #         self.save(path)
    #     else:
    #         print(f'\nLoading Weights: "{path}"\n')
    #         self.load(path)
    #         self.decoder.r = self.r.item()
    #
    # def load(self, path, device='cpu'):
    #     # because PyTorch places on CPU by default, we follow those semantics by using CPU as default.
    #     self.load_state_dict(torch.load(path, map_location=device), strict=False)
    #
    # def save(self, path):
    #     torch.save(self.state_dict(), path)
    #
    # def num_params(self, print_out=True):
    #     parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    #     if print_out:
    #         print('Trainable Parameters: %.3fM' % parameters)






class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
