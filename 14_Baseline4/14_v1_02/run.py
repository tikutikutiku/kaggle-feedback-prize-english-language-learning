import time
import pandas as pd
import numpy as np
import gc
from os.path import join as opj
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import Model, DatasetTrain, CustomCollator
from transformers import AutoTokenizer
from awp import AWP

import transformers
transformers.logging.set_verbosity_error()

discourse_type_list = [
    'Lead',
    'Position',
    'Claim',
    'Counterclaim',
    'Rebuttal',
    'Evidence',
    'Concluding Statement'
]

def run(args, trn_df, val_df, pseudo_df=None):
    output_path = opj(f'./result', args.version)
    if True:
#         if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
#             from transformers.models.deberta_v2 import DebertaV2TokenizerFast
#             tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
#             special_tokens_dict = {'additional_special_tokens': ['\n\n']}
#             _ = tokenizer.add_special_tokens(special_tokens_dict)
#         else:
#             tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
        if 'deberta-v2' in args.model or 'deberta-v3' in args.model:
            from transformers.models.deberta_v2 import DebertaV2TokenizerFast
            tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model, trim_offsets=False)
            if args.pretrained_detector_path!='none':
                special_tokens_dict = {'additional_special_tokens': ['\n\n'] + [f'[{s.upper()}]' for s in discourse_type_list]}
            else:
                special_tokens_dict = {'additional_special_tokens': ['\n\n']}
            _ = tokenizer.add_special_tokens(special_tokens_dict)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trim_offsets=False)
            if args.pretrained_detector_path!='none':
                special_tokens_dict = {'additional_special_tokens': [f'[{s.upper()}]' for s in discourse_type_list]}
                _ = tokenizer.add_special_tokens(special_tokens_dict)

        # dataset
        trn_dataset = DatasetTrain(
            trn_df, 
            tokenizer,
            mask_prob=args.mask_prob,
            mask_ratio=args.mask_ratio,
            max_length=args.max_length,
            mode=args.mode,
            aug=args.aug,
            crop_prob=args.crop_prob,
            target_cols=[args.class_name],
        )
        trn_dataloader = DataLoader(
            trn_dataset,
            batch_size=args.trn_batch_size,
            shuffle=True,
            collate_fn=CustomCollator(tokenizer),
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
        )
        
        val_dataset = DatasetTrain(
            val_df, 
            tokenizer,
            max_length=args.max_length,
            mode=args.mode,
            target_cols=[args.class_name],
        )
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
        num_train_steps = int(len(trn_dataset) / args.trn_batch_size / args.accumulate_grad_batches * args.epochs)
        model_pretraining = None
        
        if 'tascj' in args.pretrained_detector_path:
            from models_detector import TextSpanDetectorOriginal
            model_pretraining = TextSpanDetectorOriginal(
                args.model,
                num_classes=args.num_classes,
            )
            model_pretraining.load_state_dict(torch.load(args.pretrained_detector_path))
        elif args.pretrained_detector_path!='none':
            from models_detector import TextSpanDetector
            model_pretraining = TextSpanDetector(
                args.model,
                tokenizer,
                num_classes=args.num_classes,
                dynamic_positive=True,
                with_cp=False,
                rnn=args.rnn,
                loss=args.loss,
                head=args.head,
                multi_layers=args.multi_layers,
            )
            model_pretraining.load_state_dict(torch.load(args.pretrained_detector_path))
        
        model = Model(args.model, 
                      tokenizer,
                      num_labels=args.num_labels, 
                      num_labels_2=args.num_labels_2,
                      hidden_dropout_prob=args.hidden_drop_prob, 
                      p_drop=args.p_drop,
                      learning_rate=args.lr,
                      head_learning_rate=args.head_lr,
                      num_train_steps=num_train_steps,
                      warmup_ratio=args.warmup_ratio,
                      model_pretraining=model_pretraining,
                      rnn=args.rnn,
                      loss=args.loss,
                      head=args.head,
                      msd=args.msd,
                      pooling=args.pooling,
                      num_pooling_layers=args.num_pooling_layers,
                      multi_layers=args.multi_layers,
                      aug=args.aug,
                      mixup_alpha=args.mixup_alpha,
                      aug_stop_epoch=args.aug_stop_epoch,
                      p_aug=args.p_aug,
                      weight_decay=args.weight_decay,
                      freeze_layers=args.freeze_layers,
                      max_length=args.max_length,
                      mt=args.mt,
                      w_mt=args.w_mt,
                      scheduler=args.scheduler,
                      num_cycles=args.num_cycles,
                      with_cp=(args.check_pointing=='true'),
                      window_size=args.window_size,
                      inner_len=args.inner_len,
                      edge_len=args.edge_len,
                      adam_bits=args.adam_bits,
                     )
        model = model.cuda()
        if args.pretrain_path != 'none':
            model.load_state_dict(torch.load(args.pretrain_path))
            
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16=='true'))
        
        [optimizer], [scheduler] = model.configure_optimizers()
        scheduler = scheduler['scheduler']
        
        #training
        val_score_best = 1e+99
        val_loss_best  = 1e+99
        epoch_best = 0
        counter_ES = 0
        trn_score = 0
        start_time = time.time()
        for epoch in range(1, args.epochs+1):
            if epoch > args.stop_epoch:
                break
            if epoch < args.restart_epoch:
                print('epoch = ', epoch)
                for i,data in enumerate(tqdm(trn_dataloader, total=int(len(trn_dataloader)))):
                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scheduler.step()
                continue
                
            print('lr : ', [ group['lr'] for group in optimizer.param_groups ])
            
            #train
            trn_loss = 0
            trn_score = 0
            counter = 0
            tk0 = tqdm(trn_dataloader, total=int(len(trn_dataloader)))
            optimizer.zero_grad()
            trn_preds = []
            trn_trues = []
            
            if args.awp=='true':
                awp = AWP(
                    model,
                    criterion=nn.MSELoss(reduction='none'), 
                    optimizer=optimizer,
                    apex=(args.fp16=='true'),
                    adv_lr=args.awp_lr, 
                    adv_eps=args.awp_eps
                )
            
            for i,data in enumerate(tk0):
                model.train()
                batch = len(data['input_ids'])
                with torch.cuda.amp.autocast(enabled=(args.fp16=='true')):
                    pred, label, loss = model.training_step(data)
                    if args.accumulate_grad_batches > 1:
                        loss = loss / args.accumulate_grad_batches
                    scaler.scale(loss).backward()
                    
                    # Unscales the gradients of optimizer's assigned params in-place
                    #scaler.unscale_(optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                    trn_loss += loss.item() * batch * args.accumulate_grad_batches
                    
                    if args.awp=='true':
                        if epoch >= args.awp_start_epoch:
                            loss = awp.attack_backward(data)
                            scaler.scale(loss).backward()
                            awp._restore()
                            trn_loss += loss.item() * batch * args.accumulate_grad_batches

                    if (i + 1) % args.accumulate_grad_batches == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                
                trn_preds.append(pred)
                trn_trues.append(label)
                counter  += len(pred)
                tk0.set_postfix(loss=(trn_loss / counter))
                
                # eval
                if args.eval_step!=-1 and (i+1)%args.eval_step==0 and epoch>=args.eval_step_start_epoch:
                    model.eval()
                    trn_score = model.get_scores(np.vstack(trn_preds), np.vstack(trn_trues))
                    outputs = []
                    for i,data in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs.append(model.validation_step(data))
                        #release GPU memory cache
                        del data
                        torch.cuda.empty_cache()
                        gc.collect()
                    val_loss, val_score = model.validation_epoch_end(outputs)

                    #monitering
                    print('\nepoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                        epoch, 
                        trn_loss / counter,
                        val_loss, 
                        trn_score,
                        val_score
                    ))
                    if args.slack_url!='none':
                        pass
                    
                    # save
                    if args.early_stopping:
                        if val_score < val_score_best:
                            val_score_best = val_score #update
                            torch.save(model.state_dict(), 
                                       opj(output_path,f'model_{args.class_name}_seed{args.seed}_fold{args.fold}.pth')) #save
                            print(f'model_{args.class_name} (best score) saved')
                
            trn_loss = trn_loss / counter
            trn_score = model.get_scores(np.vstack(trn_preds), np.vstack(trn_trues))
            
#             # save
#             if args.mode=='pseudo':
#                 torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_epoch{epoch}_pseudo.pth')) #save
#             else:
#                 torch.save(model.state_dict(), opj(output_path,f'model_seed{args.seed}_fold{args.fold}_epoch{epoch}.pth')) #save
            
            #release GPU memory cache
            del data, loss
            torch.cuda.empty_cache()
            gc.collect()

            #eval
            model.eval()
            outputs = []
            tk1 = tqdm(val_dataloader, total=int(len(val_dataloader)))
            for i,data in enumerate(tk1):
                with torch.no_grad():
                    outputs.append(model.validation_step(data))
                #release GPU memory cache
                del data
                torch.cuda.empty_cache()
                gc.collect()
            val_loss, val_score = model.validation_epoch_end(outputs)
            
            #monitering
            print('epoch {:.0f}: trn_loss = {:.4f}, val_loss = {:.4f}, trn_score = {:.4f}, val_score = {:.4f}'.format(
                epoch,
                trn_loss,
                val_loss,
                trn_score,
                val_score
            ))
            if args.slack_url!='none':
                pass
            if epoch%10 == 0:
                print(' elapsed_time = {:.1f} min'.format((time.time() - start_time)/60))
            if args.early_stopping:
                if val_score < val_score_best:
                    val_score_best = val_score #update
                    torch.save(model.state_dict(), opj(
                        output_path,f'model_{args.class_name}_seed{args.seed}_fold{args.fold}.pth')) #save
                    print(f'model_{args.class_name} (best score) saved')
                
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print('')