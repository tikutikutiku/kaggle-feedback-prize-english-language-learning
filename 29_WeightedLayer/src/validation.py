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

from models import TARGET_COLS

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
    
    parser.add_argument("--val_batch_size", type=int, default=8, required=False)
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
    parser.add_argument("--remove_class", type=str, default="none", required=False)
    
    parser.add_argument("--use_stats", type=str, default="false", required=False)
    parser.add_argument("--use_separate_head", type=str, default="false", required=False)
    
    return parser.parse_args()

    
from models import Model, DatasetTrain, CustomCollator
    
if __name__=='__main__':
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
    train_df = pd.read_csv(opj(args.input_path, 'train.csv'))
    print('train_df.shape = ', train_df.shape)
    train_df = train_df.rename(columns={'text_id':'essay_id'})
    
    feature_cols = None
    if args.use_stats=="true":
        print("adding stats features...")
#         from enchant.checker import SpellChecker
#         import enchant
#         def get_error_count(text):
#             chkr = SpellChecker("en_US")
#             chkr.set_text(text)
#             count = 0
#             for err in chkr:
#                 count += 1
#             return count
#         def get_word_count(text):
#             return len([word for word,pos in enchant.tokenize.en.tokenize(text)])
#         def get_unique_word_count(text):
#             return len(np.unique([word for word,pos in enchant.tokenize.en.tokenize(text)]))
#         train_df["error_count"] = train_df["full_text"].apply(lambda x: get_error_count(x))
#         train_df["word_count"] = train_df["full_text"].apply(lambda x: get_word_count(x))
#         train_df["error_ratio"] = train_df["error_count"] / train_df["word_count"]
#         train_df["unique_word_count"] = train_df["full_text"].apply(lambda x: get_unique_word_count(x))
#         train_df["unique_ratio"] = train_df["unique_word_count"] / train_df["word_count"]
        feature_cols = ["error_count", "word_count", "error_ratio", "unique_word_count", "unique_ratio"]
        print("adding stats features, done")
    
    output_path = opj(f'./result', args.version)
    os.makedirs(output_path, exist_ok=True)
    fold_path = args.fold_path
    import joblib
    print('load folds...')
    trn_ids_list = joblib.load(opj(fold_path,f'trn_ids_list.joblib'))
    val_ids_list = joblib.load(opj(fold_path,f'val_ids_list.joblib'))
    
    trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold])].reset_index(drop=True)
    val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold])].reset_index(drop=True)
    #trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold][:30])].reset_index(drop=True)
    #val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold][:30])].reset_index(drop=True)
    
    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] }
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
    
    if args.class_name!='none':
        target_cols = [args.class_name]
    elif args.remove_class!='none':
        target_cols = [col for col in TARGET_COLS if col!=args.remove_class]
    else:
        target_cols = TARGET_COLS
    print("target_cols = ", target_cols)
    
    val_features = None
    if args.use_stats=='true':
        val_features = val_df[feature_cols].values
    
    val_dataset = DatasetTrain(
        val_df, 
        tokenizer,
        max_length=args.max_length,
        target_cols=target_cols,
        features=val_features if feature_cols else None,
    )
    from torch.utils.data import DataLoader
    val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
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
                  num_feats=len(feature_cols) if feature_cols else 0,
                  use_separate_head=args.use_separate_head,
                 )
    if args.weight_path!='none':
        weight_path = args.weight_path
    elif args.class_name!='none':
        weight_path = f'./result/{args.version}/model_{args.class_name}_seed{args.seed}_fold{args.fold}.pth'
    else:
        weight_path = f'./result/{args.version}/model_seed{args.seed}_fold{args.fold}.pth'
    model.load_state_dict(torch.load(weight_path))
    model = model.cuda()
    model.eval()
    
    from tqdm import tqdm
    outputs = []
    for batch_idx, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        with torch.no_grad():
            output = model.validation_step(batch)
            outputs.append(output)
    val_loss, val_score = model.validation_epoch_end(outputs)
    print('val_loss={:.4f}, val_score={:.4f}'.format(val_loss, val_score))
            
    essay_ids = []
    preds = []
    labels = []
    losses = []
    pred_seqs = []
    embeds = []
    for o in outputs:
        essay_ids.append(o['essay_id'])
        preds.append(o['pred'])
        labels.append(o['label'])
        losses.append(o['loss'])
        pred_seqs.extend(o['pred_seq'])
        embeds.append(o['embed'])

    essay_ids = np.hstack(essay_ids)    
    preds = np.vstack(preds)
    labels = np.vstack(labels)
    losses = np.vstack(losses)
    embeds = np.vstack(embeds)
    
    print('essay_ids.shape = ', essay_ids.shape)
    print('preds.shape = ', preds.shape)
    print('labels.shape = ', labels.shape)
    print('losses.shape = ', losses.shape)
    print('embeds.shape = ', embeds.shape)
    
    #np.savez_compressed(f'./result/{args.version}/embeds_{args.class_name}_fold{args.fold}.npz',embeds)
    
    pred_df = pd.DataFrame()
    pred_df['text_id'] = essay_ids
    
    if args.class_name!='none':
        pred_df[args.class_name] = preds
        pred_df[f'loss_{args.class_name}'] = losses
        pred_df.to_csv(f'./result/{args.version}/pred_{args.class_name}_fold{args.fold}.csv', index=False)
    else:
        for i,col in enumerate(target_cols):
            pred_df[col] = preds[:,i]
        for i,col in enumerate(target_cols):
            pred_df[f'loss_{col}'] = losses[:,i]
        pred_df.to_csv(f'./result/{args.version}/pred_fold{args.fold}.csv', index=False)
    
    #pred_df['pred_seq'] = pred_seqs
    