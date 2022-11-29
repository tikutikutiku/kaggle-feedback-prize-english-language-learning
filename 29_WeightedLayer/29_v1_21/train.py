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

import torch.utils.checkpoint


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
    
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--head_lr", type=float, required=True)
    parser.add_argument("--mask_prob", type=float, default=0.0, required=False)
    parser.add_argument("--mask_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--trn_batch_size", type=int, default=8, required=False)
    parser.add_argument("--val_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, required=False)
    parser.add_argument("--gradient_clip_val", type=int, default=1, required=False)
    parser.add_argument("--slack_url", type=str, default='none', required=False)
    
    parser.add_argument("--restart_epoch", type=int, default=0, required=False)
    parser.add_argument("--early_stopping", type=str, default='true', required=False)
    parser.add_argument("--patience", type=int, default=100, required=False)
    parser.add_argument("--hidden_drop_prob", type=float, default=0.1, required=False)
    parser.add_argument("--p_drop", type=float, default=0.5, required=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.0, required=False)
    
    parser.add_argument("--pretrain_path", type=str, default='none', required=False)
    parser.add_argument("--rnn", type=str, default='none', required=False)
    parser.add_argument("--head", type=str, default='simple', required=False)
    parser.add_argument("--loss", type=str, default='mse', required=False)
    parser.add_argument("--aug", type=str, default='false', required=False)
    parser.add_argument("--msd", type=str, default='false', required=False)
    parser.add_argument("--pooling", type=str, default='mean', required=False)
    parser.add_argument("--num_pooling_layers", type=int, default=1, required=False)
    parser.add_argument("--multi_layers", type=int, default=1, required=False)
    parser.add_argument("--eval_step_start_epoch", type=int, default=1, required=False)
    parser.add_argument("--eval_step", type=int, default=-1, required=False)
    parser.add_argument("--stop_epoch", type=int, default=999, required=False)
    
    parser.add_argument("--num_labels", type=int, default=6, required=False)
    parser.add_argument("--num_labels_2", type=int, default=7, required=False)
    parser.add_argument("--num_classes", type=int, default=7, required=False)
    
    parser.add_argument("--mixup_alpha", type=float, default=1.0, required=False)
    parser.add_argument("--aug_stop_epoch", type=int, default=999, required=False)
    parser.add_argument("--p_aug", type=float, default=0, required=False)
    parser.add_argument("--sampling", type=str, default='simple', required=False)
    
    parser.add_argument("--fp16", type=str, default='false', required=False)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
    parser.add_argument("--freeze_layers", type=str, default='false', required=False)
    
    parser.add_argument("--max_length", type=int, default=512, required=False)
    parser.add_argument("--preprocessed_data_path", type=str, required=False)
    parser.add_argument("--mt", type=str, default='false', required=False)
    parser.add_argument("--w_mt", type=float, default=0, required=False)
    
    parser.add_argument("--awp", type=str, default='false', required=False)
    parser.add_argument("--awp_lr", type=float, default=0, required=False)
    parser.add_argument("--awp_eps", type=float, default=0, required=False)
    parser.add_argument("--awp_start_epoch", type=int, default=0, required=False)
    
    parser.add_argument("--pretrained_detector_path", type=str, default='none', required=False)
    parser.add_argument("--scheduler", type=str, default='cosine', required=False)
    parser.add_argument("--num_cycles", type=int, default=1, required=False)
    
    parser.add_argument("--check_pointing", type=str, default='false', required=False)
    
    parser.add_argument("--window_size", type=int, default=-1, required=False)
    parser.add_argument("--inner_len", type=int, default=-1, required=False)
    parser.add_argument("--edge_len", type=int, default=-1, required=False)
    
    parser.add_argument("--adam_bits", type=int, default=32, required=False)
    
    parser.add_argument("--mode", type=str, default='train', required=False)
    #parser.add_argument("--loss_weight", type=str, default='false', required=False)
    
    parser.add_argument("--crop_prob", type=float, default=0, required=False)
    
    parser.add_argument("--class_name", type=str, default="none", required=False)
    parser.add_argument("--add_fb3", type=str, default="false", required=False)
    parser.add_argument("--fb3_fold_path", type=str, default="none", required=False)
    
    parser.add_argument("--remove_class", type=str, default="none", required=False)
    
    parser.add_argument("--use_stats", type=str, default="false", required=False)
    parser.add_argument("--use_separate_head", type=str, default="false", required=False)
    
    parser.add_argument("--low_loss_thr", type=float, default=-1, required=False)
    parser.add_argument("--high_loss_thr", type=float, default=-1, required=False)
    parser.add_argument("--oof_df_path", type=str, default='none', required=False)
    
    parser.add_argument("--last_layer_reinit", type=str, default='false', required=False)
    
    return parser.parse_args()

    
    
if __name__=='__main__':
    NUM_JOBS = 12
    args = parse_args()
    if args.seed<0:
        seed_everything(args.fold)
    else:
        seed_everything(args.fold + args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
    print('args.mode = ', args.mode)
    if args.mode=='train':
        train_df = pd.read_csv(opj(args.input_path, 'train.csv'))
        print('train_df.shape = ', train_df.shape)
    elif args.mode=='pseudo':
        train_df = pd.read_csv(args.input_path)
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
#         train_df["error_count"] = train_df["error_count"] / train_df["error_count"].max()
#         train_df["word_count"] = train_df["word_count"] / train_df["word_count"].max()
#         train_df["error_ratio"] = train_df["error_ratio"] / train_df["error_ratio"].max()
#         train_df["unique_word_count"] = train_df["unique_word_count"] / train_df["unique_word_count"].max()
#         train_df["unique_ratio"] = train_df["unique_ratio"] / train_df["unique_ratio"].max()
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
    #trn_df = train_df[train_df['essay_id'].isin(trn_ids_list[args.fold][:600])].reset_index(drop=True)
    #val_df = train_df[train_df['essay_id'].isin(val_ids_list[args.fold][:600])].reset_index(drop=True)
    
    print('trn_df.shape = ', trn_df.shape)
    print('val_df.shape = ', val_df.shape)
    
    if args.mode=='pseudo' and args.add_fb3=='true':
        print("adding fb3 data...")
        fb3_df = pd.read_csv('../../input/feedback-prize-english-language-learning/train.csv')
        fb3_df = fb3_df.rename(columns={'text_id':'essay_id'})
        fb3_trn_ids_list = joblib.load(opj(args.fb3_fold_path,f'trn_ids_list.joblib'))
        fb3_trn_df = fb3_df[fb3_df['essay_id'].isin(fb3_trn_ids_list[args.fold])].reset_index(drop=True)
        
        print("trn_df.columns = ", trn_df.columns)
        print("fb3_trn_df.columns = ", fb3_trn_df.columns)
        print("fb3_trn_df.shape = ", fb3_trn_df.shape)
        print("fb3_trn_df[trn_df.columns].shape = ", fb3_trn_df[trn_df.columns].shape)
        
        trn_df = pd.concat([trn_df, fb3_trn_df[trn_df.columns]], axis=0).reset_index(drop=True)
        print("adding fb3 data, done")
        print('trn_df.shape = ', trn_df.shape)
        
    if args.low_loss_thr != -1:
        oof_df = pd.read_csv(args.oof_df_path)
        low_loss_idxs = oof_df.loc[oof_df[f'loss_{args.class_name}']<args.low_loss_thr, 'text_id'].values
        trn_df = trn_df[trn_df['essay_id'].isin(low_loss_idxs)].reset_index(drop=True)
        print('trn_df.shape = ', trn_df.shape)
        
    if args.high_loss_thr != -1:
        oof_df = pd.read_csv(args.oof_df_path)
        high_loss_idxs = oof_df.loc[oof_df[f'loss_{args.class_name}']>args.high_loss_thr, 'text_id'].values
        trn_df = trn_df[trn_df['essay_id'].isin(high_loss_idxs)].reset_index(drop=True)
        print('trn_df.shape = ', trn_df.shape)
    
    from run import run
    run(args, trn_df, val_df, pseudo_df=None, feature_cols=feature_cols)