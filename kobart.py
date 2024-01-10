
import tempfile
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from transformers import (BartForConditionalGeneration,
                          PreTrainedTokenizerFast)
from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM





logger = logging.getLogger()
logger.setLevel(logging.INFO)

args={'checkpoint_path':'/content/drive/MyDrive/Capstone_pj',
         'chat':False,
         'train_file':'/content/drive/MyDrive/Capstone_pj/double_train.csv',
         'test_file':'/content/drive/MyDrive/Capstone_pj/dev.csv',
        'tokenizer_path':'tokenizer',
         'batch_size':14,
         'max_seq_len':36,
         'num_workers':5,
         'batch_size':14,
        'gpus':1,
        'num_nodes':1,
         'lr':5e-5,
      'max_epochs':1,
         'warmup_ratio':0.1,
         'model_path': None}

class KoBARTConditionalGeneration(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(KoBARTConditionalGeneration, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained( "gogamza/kobart-base-v2", bos_token="<s>", eos_token="</s>",unk_token='<unk>',pad_token='<pad>',mask_token='<mask>')

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args['lr'], correct_bias=False)
        # warm up lr
        num_workers = (self.args['gpus'] if self.args['gpus'] is not None else 1) * (self.args['num_nodes'] if self.args['num_nodes'] is not None else 1)
        data_len = len(train.dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.args['batch_size'] * num_workers) * self.args['max_epochs'])
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.args['warmup_ratio'])
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def forward(self, inputs):
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)


    def chat(self, text):
        input_ids =  [self.tokenizer.bos_token_id] + self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        res_ids = self.model.generate(torch.tensor([input_ids]),
                                            
                                            num_beams=5,
                                            eos_token_id=self.tokenizer.eos_token_id,
                                            bad_words_ids=[[self.tokenizer.unk_token_id]])
        a = self.tokenizer.batch_decode(res_ids.tolist())[0]
        return a.replace('<s>', '').replace('</s>', '')
    


def kobart(trans,model_path):
   args={'checkpoint_path':'/content/drive/MyDrive/Capstone_pj',
         'chat':False,
         'train_file':'/content/drive/MyDrive/Capstone_pj/double_train.csv',
         'test_file':'/content/drive/MyDrive/Capstone_pj/dev.csv',
        'tokenizer_path':'tokenizer',
         'batch_size':14,
         'max_seq_len':36,
         'num_workers':5,
         'batch_size':14,
        'gpus':1,
        'num_nodes':1,
         'lr':5e-5,
      'max_epochs':1,
         'warmup_ratio':0.1,
         'model_path': None}
   model = KoBARTConditionalGeneration(args)
   checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
   model.load_state_dict(checkpoint['state_dict'])
   trans=model.chat(trans)
   print(trans)
   return trans