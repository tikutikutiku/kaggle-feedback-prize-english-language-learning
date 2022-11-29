import torch
print(torch.__name__, torch.__version__)

import argparse
import os
from os.path import join as opj
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

#from models import TARGET_COLS

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_path", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--seed", type=int, default=-1, required=True)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-english-language-learning/', required=False)
    
    parser.add_argument("--test_batch_size", type=int, default=8, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    
    parser.add_argument("--pretrain_path", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    parser.add_argument("--pooling", type=str, default='mean', required=False)
    parser.add_argument("--num_pooling_layers", type=int, default=1, required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    
    parser.add_argument("--num_labels", type=int, default=3, required=False)
    parser.add_argument("--num_labels_2", type=int, default=7, required=False)
    
    parser.add_argument("--l2norm", type=str, default='false', required=False)
    
    parser.add_argument("--max_length", type=int, default=512, required=False)
    parser.add_argument("--preprocessed_data_path", type=str, required=False)
    
    parser.add_argument("--mt", type=str, default='false', required=False)
    parser.add_argument("--weight_path", type=str, default='none', required=False)
    
    parser.add_argument("--window_size", type=int, default=512, required=False)
    parser.add_argument("--inner_len", type=int, default=384, required=False)
    parser.add_argument("--edge_len", type=int, default=64, required=False)
    
    parser.add_argument("--class_name", type=str, default="none", required=False)
    
    parser.add_argument("--unlabeled_data_path", type=str, required=False)
    
    return parser.parse_args()

    
from models import Model, DatasetTest, CustomCollator
    
if __name__=='__main__':
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
    test_df = pd.read_csv(args.unlabeled_data_path)
    print('test_df.shape = ', test_df.shape)
    test_df = test_df.rename(columns={'id':'essay_id'})
    
    if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] }
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
    
    print('test_df.columns = ', test_df.columns)
    
    from torch.utils.data import DataLoader
    test_dataset = DatasetTest(
        test_df,
        tokenizer, 
        max_length=args.max_length,
    )
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=False,
        )
    
    #model
    model = Model(args.model, 
                  tokenizer,
                  num_labels=args.num_labels, 
                  num_labels_2=args.num_labels_2,
                  rnn=args.rnn,
                  loss=args.loss,
                  head=args.head,
                  pooling=args.pooling,
                  num_pooling_layers=args.num_pooling_layers,
                  multi_layers=args.multi_layers,
                  l2norm=args.l2norm,
                  max_length=args.max_length,
                  mt=args.mt,
                  window_size=args.window_size,
                  inner_len=args.inner_len,
                  edge_len=args.edge_len,
                 )
    if args.weight_path!='none':
        weight_path = args.weight_path
    else:
        weight_path = f'./result/{args.version}/model_{args.class_name}_seed{args.seed}_fold{args.fold}.pth'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    
    from tqdm import tqdm
    outputs = []
    for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        with torch.no_grad():
            output = model.test_step(batch)
            outputs.append(output)
            
    essay_ids = []
    preds = []
    #pred_seqs = []
    for o in outputs:
        essay_ids.append(o['essay_id'])
        preds.append(o['pred'])
        #pred_seqs.extend(o['pred_seq'])

    essay_ids = np.hstack(essay_ids)    
    preds = np.vstack(preds)
    print('essay_ids.shape = ', essay_ids.shape)
    print('preds.shape = ', preds.shape)
    
    pred_df = pd.DataFrame()
    pred_df['text_id'] = essay_ids
    pred_df[args.class_name] = preds
    
    #pred_df['pred_seq'] = pred_seqs
    
    pred_df.to_csv(f'./result/{args.version}/pseudo_{args.class_name}_fold{args.fold}.csv', index=False)