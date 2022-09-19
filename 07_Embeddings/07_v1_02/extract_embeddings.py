from models import to_gpu, CustomCollator
import torch
import torch    
import numpy as np
from torch.utils.data import Dataset

MAX_LENGTH = 512 
BS = 8
NUM_WORKERS = 4
    
class DatasetFB3(Dataset):
    def __init__(self, df, tokenizer, max_length=-1):
        self.df = df
        self.unique_ids = sorted(df['text_id'].unique())
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.unique_ids)
    
    def __getitem__(self, idx):
        text_id = self.unique_ids[idx]
        sample_df = self.df[self.df['text_id']==text_id].reset_index(drop=True)
        
        text = sample_df['full_text'].values[0]
        if self.max_length==-1:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True)
        else:
            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, truncation=True)
        input_ids = torch.LongTensor(tokens['input_ids'])
        attention_mask = torch.ones(len(input_ids)).long()
        
        return dict(
            text_id = text_id,
            text = text,
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    

def get_embeddings(model_name, train_df, pretrain_path):
    if 'deberta-v2' in model_name or 'deberta-v3' in model_name:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name, trim_offsets=False)
        special_tokens_dict = {'additional_special_tokens': ['\n\n'] }
        _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trim_offsets=False)

    dataset_fb3 = DatasetFB3(
            train_df,
            tokenizer,
            max_length=MAX_LENGTH,
        )
    from torch.utils.data import DataLoader
    dataloader_fb3 = DataLoader(
                dataset_fb3,
                batch_size=BS,
                shuffle=False,
                collate_fn=CustomCollator(tokenizer),
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
            )

    if 'tascj' in args.pretrain_path:
        from models_detector import TextSpanDetectorOriginal
        model = TextSpanDetectorOriginal(
            model_name,
            num_classes=7,
        )
        model.load_state_dict(torch.load(pretrain_path))
        model = model.model.deberta
    elif args.pretrain_path!='none':
        from models_detector import TextSpanDetector
        model = TextSpanDetector(
            model_name,
            tokenizer,
            num_classes=7,
        )
        model.load_state_dict(torch.load(pretraine_path))
        model = model.transformer
    else:
        from transformers import AutoConfig, AutoModel
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": False,
            }
        )
        model = AutoModel.from_pretrained(model_name, config=model_config)
    
    model = model.cuda()
    model.eval()

    from tqdm import tqdm
    embeds = []
    for batch_idx, batch in tqdm(enumerate(dataloader_fb3), total=len(dataloader_fb3)):
        batch = to_gpu(batch)
        with torch.no_grad():
            hidden_states = model(input_ids=batch['input_ids'],
                                  attention_mask=batch['attention_mask']).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            mask_sum = torch.sum(mask, dim=1)
            embed = torch.sum(hidden_states*mask, dim=1) / mask_sum # (bs, hidden_size*multi_layers)
            embed = embed.detach().cpu().numpy()
            embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
            embeds.append(embed)

    embeds = np.vstack(embeds)
    print('embeds.shape = ', embeds.shape)

    del model
    torch.cuda.empty_cache()
    
    return embeds




import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_path", type=str, default='../../input/feedback-prize-english-language-learning/', required=False)
    parser.add_argument("--pretrain_path", type=str, default='none', required=False)
    
    return parser.parse_args()
    
    
if __name__=='__main__':
    args = parse_args()
    print('model_name = ', args.model_name)
    
    import pandas as pd
    from os.path import join as opj
    train_df = pd.read_csv(opj(args.input_path,'train.csv'))
    
    embed = get_embeddings(args.model_name, train_df, args.pretrain_path)
    print('embed.shape = ', embed.shape)
    import os
    os.makedirs(f'./result', exist_ok=True)
    os.makedirs(f'./result/{args.version}', exist_ok=True)
    model_name = args.model_name.split('/')[-1]
    np.savez_compressed(f'./result/{args.version}/embed_{model_name}.npz', embed)