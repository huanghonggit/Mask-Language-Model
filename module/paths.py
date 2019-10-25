import os


class Paths:
    def __init__(self, output_path):
        self.output_path = output_path
        self.bert_path = f'{output_path}/model_bert'
        self.mlm_path = f'{output_path}/model_mlm'
        self.plt_train_attn_path = f'{output_path}/train_plt_attn'
        self.plt_valid_attn_path = f'{output_path}/valid_plt_attn'
        self.bert_checkpoints_path = f'{output_path}/bert_checkpoints_path'
        self.runs_path = f'{output_path}/runs'
        self.create_paths()

    def create_paths(self):
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.bert_path, exist_ok=True)
        os.makedirs(self.mlm_path, exist_ok=True)
        os.makedirs(self.plt_train_attn_path, exist_ok=True)
        os.makedirs(self.plt_valid_attn_path, exist_ok=True)
        os.makedirs(self.bert_checkpoints_path, exist_ok=True)
        os.makedirs(self.runs_path, exist_ok=True)


