
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override input/outputpath in __main__.py)
input_path = ''
output_path = ''
save_runs_path = ''


# Mlm params ------------------------------------------------------------------------------------------------------#
embed_dim = 256
hidden = 256
attn_layers = 4
attn_heads = 4
enc_maxlen = 1024
pos_dropout_rate = 0.1
enc_conv1d_dropout_rate = 0.2
enc_conv1d_layers = 3
enc_conv1d_kernel_size = 5
enc_ffn_dropout_rate = 0.1

self_att_dropout_rate = 0.1
self_att_block_res_dropout = 0.1


# Optimizer params -----------------------------------------------------------------------------------------------------#
lr = 0.0001
adam_beta1 = 0.9
adam_beta2 = 0.9
adam_weight_decay = 0
mlm_clip_grad_norm = 1.0
clip_grad_norm = True
warmup = 0.1


# Trainer params -----------------------------------------------------------------------------------------------------#
epochs = 10
log_freq = 10
save_train_loss = 50
save_valid_loss = 500
save_model = 10000
save_checkpoint = 10000
save_runs = 500
batch_size = 96
valid_size = 96


# ------------------------------------------------------------------------------------------------------------------#

