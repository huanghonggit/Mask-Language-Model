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
        # pos = _add_pos_embedding(x)
        pos = self.pos_emb(pos)
        x = pos * self.alpha + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)  # x:32,147,256  为每一层都加了个feedforward
            attns.append(attn)  # 128,147,147

        return x, attns


    # def init_model(self):
    #     for p in self.parameters():
    #         if p.dim() > 1: nn.init.xavier_uniform_(p)
    #     print("初始化模型参数！")

    # def get_step(self):
    #     return self.step.data.item()
    #
    # def reset_step(self):
    #     self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    def checkpoint(self, path, steps):
        # k_steps = self.get_step() // 1000
        # self.save(f'{path}/mlm_checkpoint_{steps}k_steps.pyt')
        self.save(f'{path}/mlm_checkpoint_{steps}steps.pyt')



        # self.pos_emb = nn.Embedding(args.char_maxlen, embed_dim, padding_idx=0)  # 512,32

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

    # def load(self, path, device='cpu'):
    #     # because PyTorch places on CPU by default, we follow those semantics by using CPU as default.
    #     self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path):
        torch.save(self.state_dict(), path)
    #
    # def num_params(self, print_out=True):
    #     parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    #     if print_out:
    #         print('Trainable Parameters: %.3fM' % parameters)

# class BERT(nn.Module):
#     """
#     BERT model : Bidirectional Encoder Representations from Transformers.
#     """
#
#     def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
#         """
#         :param vocab_size: vocab_size of total words
#         :param hidden: BERT model hidden size
#         :param n_layers: numbers of Transformer blocks(layers)
#         :param attn_heads: number of attention heads
#         :param dropout: dropout rate
#         """
#
#         super().__init__()
#         self.hidden = hidden
#         self.n_layers = n_layers
#         self.attn_heads = attn_heads
#
#         # paper noted they used 4*hidden_size for ff_network_hidden_size
#         self.feed_forward_hidden = hidden * 4
#
#         # embedding for BERT, sum of positional, segment, token embeddings
#         self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
#
#         # multi-layers transformer blocks, deep network
#         self.transformer_blocks = nn.ModuleList(
#             [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
#
#     def forward(self, x):
#         # attention masking for padded token
#         # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
#         # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1) # x：4，512-->4,1,512,512
#         c_mask = x.ne(0).type(torch.float)
#         mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
#
#         # embedding the indexed sequence to sequence of vectors
#         x = self.embedding(x)
#
#         # running over multiple transformer blocks
#         attn_list = []
#         for transformer in self.transformer_blocks:
#             x, atten = transformer.forward(x, mask, c_mask)
#             attn_list.append(atten)
#
#         return x, attn_list
