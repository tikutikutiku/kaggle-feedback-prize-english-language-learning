import torch

def to_gpu(data):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(data, dict):
        return {k: to_gpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_gpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    else:
        return data


def to_np(t):
    '''
    https://www.kaggle.com/code/tascj0/a-text-span-detector
    '''
    if isinstance(t, torch.Tensor):
        return t.data.cpu().numpy()
    else:
        return t


import torch
from torch import nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoModel, AutoTokenizer, AdamW,
                          get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup,
                          get_linear_schedule_with_warmup)
import bitsandbytes as bnb
from metrics import calc_metric
TARGET_COLS = ['cohesion','syntax','vocabulary','phraseology','grammar','conventions']

class ResidualLSTM(nn.Module):
    '''Based on Shujun's code'''
    def __init__(self, d_model, rnn='GRU'):
        super(ResidualLSTM, self).__init__()
        self.downsample=nn.Linear(d_model,d_model//2)
        if rnn=='GRU':
            self.LSTM=nn.GRU(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM=nn.LSTM(d_model//2, d_model//2, num_layers=2, bidirectional=False, dropout=0.2)
        self.linear=nn.Linear(d_model//2, d_model)
        self.norm= nn.LayerNorm(d_model)
    def forward(self, x):
        res=x
        x=self.downsample(x)
        x, _ = self.LSTM(x)
        x = self.linear(x)
        x=res+x
        return self.norm(x)
        
        
# based on https://www.kaggle.com/gogo827jz/roberta-model-parallel-fold-training-on-tpu
class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        #context_vector = torch.sum(context_vector, dim=1)
        return context_vector
        
        
class Model(nn.Module):
    def __init__(self, 
                 model_name, 
                 tokenizer,
                 num_labels, 
                 num_labels_2,
                 hidden_dropout_prob=0, 
                 learning_rate=1e-5,
                 head_learning_rate=1e-3,
                 num_train_steps=0,
                 p_drop=0,
                 warmup_ratio = 0,
                 model_pretraining=None,
                 rnn='none',
                 loss='mse',
                 head='simple',
                 msd='false',
                 multi_layers=1,
                 aug='none',
                 mixup_alpha=1.0,
                 aug_stop_epoch=999,
                 p_aug=0,
                 weight_decay=0.01,
                 freeze_layers='false',
                 mt='false',
                 w_mt=0,
                 scheduler='cosine',
                 num_cycles=1,
                 with_cp=False,
                 window_size=-1,
                 inner_len=-1,
                 edge_len=-1,
                 adam_bits=32,
                 **kwargs,
                ):
        super().__init__()
        self._current_epoch = 1
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.hidden_dropout_prob = hidden_dropout_prob
        self.warmup_ratio = warmup_ratio 
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.num_labels_2 = num_labels_2
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.loss = loss
        self.msd = msd
        self.multi_layers = multi_layers
        self.aug = aug
        self.mixup_alpha = mixup_alpha
        self.aug_stop_epoch = aug_stop_epoch
        self.p_aug = p_aug
        self.weight_decay = weight_decay
        self.mt = mt
        self.w_mt = w_mt
        self.scheduler = scheduler
        self.num_cycles = num_cycles
        self.window_size = window_size
        self.inner_len = inner_len
        self.edge_len = edge_len
        self.adam_bits = adam_bits
        self.last_hidden_only = False
        
        if model_pretraining is not None:
            try:
                self.transformer = model_pretraining.transformer
                self.config = model_pretraining.config
            except:
                self.last_hidden_only = True
                self.multi_layers = 1
                self.transformer = model_pretraining.model.deberta
                self.config = AutoConfig.from_pretrained(model_name)
                self.config.update(
                    {
                        "output_hidden_states": True,
                        "hidden_dropout_prob": self.hidden_dropout_prob,
                        "add_pooling_layer": False,
                        "num_labels": self.num_labels,
                    }
                )
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.config.update(
                {
                    "output_hidden_states": True,
                    "hidden_dropout_prob": self.hidden_dropout_prob,
                    "add_pooling_layer": False,
                    "num_labels": self.num_labels,
                }
            )
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
            
        # resize
        self.transformer.resize_token_embeddings(len(tokenizer))
        
        if with_cp:
            self.transformer.gradient_checkpointing_enable()
            
        # freeze some layers for large models
        if freeze_layers == 'true':
            if 'deberta-v2-xxlarge' in model_name:
                print('freeze 24/48')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:24].requires_grad_(False) # freeze 24/48
            elif 'deberta-v2-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
            elif 'deberta-xlarge' in model_name:
                print('freeze 12/24')
                self.transformer.embeddings.requires_grad_(False)
                self.transformer.encoder.layer[:12].requires_grad_(False) # freeze 12/24
        
        if rnn=='none':
            self.rnn = nn.Identity()
        elif rnn=='lstm':
            self.rnn = ResidualLSTM(self.config.hidden_size*self.multi_layers, rnn='LSTM')
        elif rnn=='gru':
            self.rnn = ResidualLSTM(self.config.hidden_size*self.multi_layers, rnn='GRU')
        else:
            raise Exception()
    
        if self.msd=='true':
            self.dropout_1 = nn.Dropout(0.1)
            self.dropout_2 = nn.Dropout(0.2)
            self.dropout_3 = nn.Dropout(0.3)
            self.dropout_4 = nn.Dropout(0.4)
            self.dropout_5 = nn.Dropout(0.5)
            
        if head=='simple':
            self.head = nn.Sequential(
                #nn.Dropout(p_drop),
                nn.LayerNorm(self.config.hidden_size*self.multi_layers),
                nn.Linear(self.config.hidden_size*self.multi_layers, self.num_labels)
            )
        elif head=='attention':
            n_hidden = self.config.hidden_size*self.multi_layers
            self.head = nn.Sequential(
                nn.LayerNorm(n_hidden),
                AttentionBlock(n_hidden,n_hidden,1),
                nn.Linear(n_hidden, self.num_labels)
            )
        elif head=='custom1':
            self.head = nn.Sequential(
                nn.LayerNorm(self.config.hidden_size*self.multi_layers),
                nn.Linear(self.config.hidden_size*self.multi_layers, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, self.num_labels),
            )
        else:
            raise Exception()
        self._init_weights(self.head)
        
        # for multi-task
        if self.mt=='true':
            self.head2 = nn.Sequential(
                    nn.Dropout(p_drop),
                    nn.Linear(self.config.hidden_size*self.multi_layers, self.num_labels_2)
                )
            self._init_weights(self.head2)

        if loss=='mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss=='l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss=='smoothl1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        elif loss=='bce':
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        #elif loss=='xentropy':
        #    self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            raise Exception()
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward_logits(self, input_ids, attention_mask, aug=False, save_pred_seq=False, return_embed=False):
        assert self.multi_layers==1
        
        # sliding window approach to deal with longer tokens than max_length
        # https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313235
        L = input_ids.size(1)
        if self.window_size==-1 or L <= self.window_size:
            x = self.transformer(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state
        else:
            #assert len(input_ids)==1
            segments = (L - self.window_size) // self.inner_len
            if (L - self.window_size) % self.inner_len > self.edge_len:
                segments += 1
            elif segments == 0:
                segments += 1
            x = self.transformer(input_ids=input_ids[:,:self.window_size],
                                 attention_mask=attention_mask[:,:self.window_size]).last_hidden_state
            for i in range(1,segments+1):
                start = self.window_size - self.edge_len + (i-1)*self.inner_len
                end   = self.window_size - self.edge_len + (i-1)*self.inner_len + self.window_size
                end = min(end, L)
                x_next = self.transformer(input_ids=input_ids[:,start:end],
                                          attention_mask=attention_mask[:,start:end]).last_hidden_state
                if i==segments:
                    x_next = x_next[:,self.edge_len:]
                else:
                    x_next = x_next[:,self.edge_len:self.edge_len+self.inner_len]
                x = torch.cat([x,x_next], dim=1)
                
        # apply rnn
        hidden_states = self.rnn(x) # (bs,num_tokens,hidden_size*multi_layers)

        # pooling
        mask = attention_mask.unsqueeze(-1)
        mask_sum = torch.sum(mask, dim=1)
        logits = torch.sum(hidden_states*mask, dim=1) / mask_sum # (bs, hidden_size*multi_layers)
        embed = logits.detach().cpu().numpy() 
        embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
        
        if save_pred_seq:
            pred_seq = []
        
        if aug:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch_size = logits.size()[0]
            index = torch.randperm(batch_size).cuda()
            logits = lam * logits + (1 - lam) * logits[index, :]
        
        if self.msd=='true' and self.training:
            logits1_1 = self.head(self.dropout_1(logits))
            logits1_2 = self.head(self.dropout_2(logits))
            logits1_3 = self.head(self.dropout_3(logits))
            logits1_4 = self.head(self.dropout_4(logits))
            logits1_5 = self.head(self.dropout_5(logits))
            logits1 = (logits1_1 + logits1_2 + logits1_3 + logits1_4 + logits1_5) / 5.0
        else:
            logits1 = self.head(logits) # (bs,num_labels)
            
        if return_embed:
            if save_pred_seq:
                return logits1, pred_seq, embed
            else:
                return logits1, embed
        else:
            if save_pred_seq:
                return logits1, pred_seq
            else:
                return logits1
    
    
    def logits_fn(self, *wargs, **kwargs):
        if self.mt=='true':
            logits,_ = self.forward_logits(**kwargs)
        else:
            logits = self.forward_logits(**kwargs)
        return logits


    def training_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
            'aug':False,
        }
        
        # get loss
        if self.loss in ['mse','l1','smoothl1']:
            if self.mt=='true' and self.training:
                logits, logits2 = self.forward_logits(**input_data)
                loss = self.get_losses(logits, data['label']).mean()
                loss += self.w_mt * self.get_losses2(logits2, data['label2']).mean()
            elif self.aug=='mixup' and np.random.random()<self.p_aug and self._current_epoch<self.aug_stop_epoch and self.training:
                input_data['aug'] = True
                logits, index, lam = self.forward_logits(**input_data)
                loss_a = self.get_losses(logits, data['label']).mean()
                loss_b = self.get_losses(logits, data['label'][index]).mean()
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                logits = self.forward_logits(**input_data)
                loss = self.get_losses(logits.reshape(-1,self.num_labels), data['label'].reshape(-1,self.num_labels)).mean()
        else:
            raise NotImplementedError()

        pred = logits.detach().cpu().numpy() # (bs,num_labels)
        label = data['label'].detach().cpu().numpy()
            
        return pred, label, loss
    
    def validation_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
            'aug':False,
            'return_embed':True,
        }
        
        # get loss
        if self.loss in ['mse','l1','smoothl1']:
            input_data.update({'save_pred_seq':True})
            logits, pred_seq, embed = self.forward_logits(**input_data)
            loss = self.get_losses(
                logits.reshape(-1,self.num_labels),
                data['label'].reshape(-1,self.num_labels)
            ).detach().cpu().numpy()
        else:
            raise NotImplementedError()
            
        # get pred
        pred = logits.detach().cpu().numpy().reshape(-1,self.num_labels)
            
        output = {
            'loss':loss,
            'pred':pred,
            'label':data['label'].detach().cpu().numpy(),
            'text':data['text'],
            'essay_id':data['essay_id'],
            'embed':embed,
        }
        if input_data['save_pred_seq']:
            output.update({'pred_seq':pred_seq})
        return output
    
    def validation_epoch_end(self, outputs):
        losses = []
        preds = []
        labels = []
        for o in outputs:
            losses.append(o['loss'])
            preds.append(o['pred'])
            labels.append(o['label'])

        losses = np.vstack(losses).mean()
        preds = np.vstack(preds)
        labels = np.vstack(labels)
        
        scores = self.get_scores(preds, labels)
        self._current_epoch += 1
        return losses, scores
    
    def test_step(self, batch):
        data = to_gpu(batch)
        input_data = {
            'input_ids':data['input_ids'],
            'attention_mask':data['attention_mask'],
            'aug':False,
        }
        logits = self.forward_logits(**input_data)
        pred = logits.softmax(dim=-1).detach().cpu().numpy().reshape(-1,self.num_labels)
        return {
            'pred':pred,
            'text':data['text'],
            'essay_id':data['essay_id'],
        }
        
    def configure_optimizers(self):
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler(optimizer)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
    
    def fetch_optimizer(self):
        if self.rnn!='none':
            head_params = list(self.head.named_parameters()) + list(self.rnn.named_parameters())
        else:
            head_params = list(self.head.named_parameters())
        param_optimizer = list(self.transformer.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n,p in head_params], 
                "weight_decay": 0.01,
                "lr": self.head_learning_rate,
            },
        ]
        if self.mt=='true':
            head2_params = list(self.head2.named_parameters())
            optimizer_parameters.append(
                {
                    "params": [p for n,p in head2_params], 
                    "weight_decay": 0.01,
                    "lr": self.head_learning_rate,

                },
            )
        #optimizer = AdamW(optimizer_parameters, lr=self.learning_rate)
        
        # https://www.kaggle.com/code/nbroad/8-bit-adam-optimization/notebook
        # These are the only changes you need to make
        # The first part sets the optimizer to use 8-bits
        # The for loop sets embeddings to use 32-bits
        if self.adam_bits == 32:
            optimizer = bnb.optim.AdamW32bit(optimizer_parameters, lr=self.learning_rate)
        if self.adam_bits == 8:
            optimizer = bnb.optim.AdamW8bit(optimizer_parameters, lr=self.learning_rate)
            # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
            for module in self.transformer.modules():
                if isinstance(module, torch.nn.Embedding):
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        module, 'weight', {'optim_bits': 32}
                    )   
        
        return optimizer

    def fetch_scheduler(self, optimizer):
        print('self.warmup_ratio = ', self.warmup_ratio)
        print('self.num_train_steps = ', self.num_train_steps)
        print('num_warmup_steps = ', int(self.warmup_ratio * self.num_train_steps))
        if self.scheduler=='cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=0.5*self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='cosine_hard':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                num_cycles=self.num_cycles,
                last_epoch=-1,
            )
        elif self.scheduler=='linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.warmup_ratio * self.num_train_steps),
                num_training_steps=self.num_train_steps,
                last_epoch=-1,
            )
        return scheduler
    
    def get_losses(self, logits, label):
        loss = self.loss_fn(logits, label)
        return loss
    
    def get_losses2(self, logits, label):
        raise NotImplementedError()
    
    def get_scores(self, pred, label):
        score = calc_metric(pred=pred, gt=label)
        return score
    
import numpy as np
from torch.utils.data import Dataset

class DatasetTrain(Dataset):
    def __init__(self, df, tokenizer, mask_prob=0, mask_ratio=0, max_length=-1, crop_prob=0, 
                 aug='false', mode='train'):
        self.df = df
        self.unique_ids = sorted(df['essay_id'].unique())
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_ratio = mask_ratio
        self.aug = aug
        self.mode = mode
        self.target_cols = TARGET_COLS
        self.max_length = max_length
        self.crop_prob = crop_prob
        
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        essay_id = self.unique_ids[idx]
        sample_df = self.df[self.df['essay_id']==essay_id].reset_index(drop=True)
        
        text = sample_df['full_text'].values[0]
        labels = sample_df[self.target_cols].values[0]
        
        if self.max_length==-1:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
        else:
            if np.random.random() < self.crop_prob:
                pre_tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
                tokens = {}
                start_token = np.random.randint(1, max(2, len(pre_tokens['input_ids'])-self.max_length ) )
                max_len = min(self.max_length, len(pre_tokens['input_ids']))
                for k,v in pre_tokens.items():
                    tokens[k] = [v[0]] + v[start_token:start_token+max_len-2] + [v[-1]]
            else:
                tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, 
                                                    max_length=self.max_length, truncation=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        # random masking
        if np.random.random() < self.mask_prob:
            all_inds = np.arange(1, len(input_ids)-1)
            n_mask = max(int(len(all_inds) * self.mask_ratio), 1)
            np.random.shuffle(all_inds)
            mask_inds = np.array([inds for inds in all_inds[:n_mask]])
            input_ids[mask_inds] = self.tokenizer.mask_token_id
        
        output = dict(
            essay_id = essay_id,
            text = text,
            input_ids = input_ids,
            attention_mask = attention_mask,
            label = labels,
        )
        return output
    
    
class DatasetTest(Dataset):
    def __init__(self, df, tokenizer, max_length=-1):
        self.df = df
        self.unique_ids = sorted(df['essay_id'].unique())
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        essay_id = self.unique_ids[idx]
        sample_df = self.df[self.df['essay_id']==essay_id].reset_index(drop=True)
        
        text = sample_df['full_text'].values[0]
        if self.max_length==-1:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
        else:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        return dict(
            essay_id = essay_id,
            text = text,
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    
class CustomCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.target_cols = TARGET_COLS
        
    def __call__(self, samples):
        output = dict()
        
        for k in samples[0].keys():
            output[k] = [sample[k] for sample in samples]
        
        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])
        
        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s.tolist() + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist() for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s.tolist() for s in output["attention_mask"]]
            
        output["input_ids"] = torch.LongTensor(output["input_ids"])
        output["attention_mask"] = torch.LongTensor(output["attention_mask"])

        if 'label' in output.keys():
            output['label'] = torch.FloatTensor(output['label'])
        return output