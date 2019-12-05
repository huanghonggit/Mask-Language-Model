import torch.nn as nn
import config.hparams as hp
import torch
from model.transfomer_block import EncoderPrenet
from model.transfomer_block import get_sinusoid_encoding_table, Attention, FFN, clones


class BERT(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, embed_dim, hidden, args=None):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(BERT, self).__init__()
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.char_nums = args.char_nums
        self.char_maxlen = hp.enc_maxlen
        self.attn_layers = hp.attn_layers
        self.attn_heads = hp.attn_heads

        self.alpha = nn.Parameter(torch.ones(1))
        self.embed = nn.Embedding(self.char_nums, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.char_maxlen, hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=hp.pos_dropout_rate)
        self.encoder_prenet = EncoderPrenet(embed_dim, hidden)
        self.layers = clones(Attention(hidden, self.attn_heads), self.attn_layers)
        self.ffns = clones(FFN(hidden), self.attn_layers)

        # self.init_model()
        # self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def forward(self, x, pos):

        if self.training:
            c_mask = x.ne(0).type(torch.float)
            mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        x = self.embed(x)           # B*T*d
        x = self.encoder_prenet(x)  # 三个卷积

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, attns

    def checkpoint(self, path, steps):
        self.save(f'{path}/mlm_checkpoint_{steps}steps.pyt')

    def save(self, path):
        torch.save(self.state_dict(), path)
