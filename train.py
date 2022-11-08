# train
# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from tokenizers import AddedToken #XXX
from utils.utils_data import BYTE_TOKENS #XXX
import argparse
from utils.data_utils import *
from utils.distributed_utils import *
from utils.utils import *
from utils.metrics import *
from model.model import *

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_name', type=str, help = 'test_name')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    # data
    parser.add_argument('--train_data', type=str, help = 'train_data 위치')
    parser.add_argument('--val_data', type=str, help='val data 위치')
    # logging 관련
    parser.add_argument('--logging_term', type=int, default = 100)
    
    # 학습 관련
    parser.add_argument('--epochs', default = 10, type=int)
    parser.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    parser.add_argument('--batch_size', default = 8, type=int)
    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--warmup', type=float, default = 0.05)
    parser.add_argument('--decay', type=float, default = 0.05)
    parser.add_argument('--fp16', type=str2bool, default = False)
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    # model
    parser.add_argument('--top_n', type=int, default = 1)
    parser.add_argument('--contain_title', type=str2bool, default=True)
    parser.add_argument('--answer_max_length', type=int)
    # specific
    parser.add_argument('--is_train', type=str2bool, default = False ,help = 'mode')
    parser.add_argument('--shuffle', type=str2bool, default = False ,help = 'mode')
    parser.add_argument('--include_history', type=str2bool, help = 'include history')
    parser.add_argument('--just_user', type=str2bool, help = 'include history')
    parser.add_argument('--history_turn', type=int, help = 'how many turn do you want to include in history')

    # distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    parser.add_argument('--early_stop', type=str2bool, default = True) # XXX220919
    parser.add_argument('--patience', type=int, default = 3)
    args  = parser.parse_args()
    return args

def make_optimizer_group(args, model):
    decay = args.decay
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
    'params': [
        p for n, p in param_optimizer
        if not any(nd in n for nd in no_decay)
    ],
    'weight_decay':
    decay
      }, {
    'params':
    [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    'weight_decay':
    0.0
    }]
    return optimizer_grouped_parameters

def get_scheduler(args, train_dataloader):
    total_step = len(train_dataloader)*args.epochs
    warmup = total_step * args.warmup
    linear_scheduler = lambda step: min(1/warmup*step,1.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = linear_scheduler)
    return scheduler

# evaluation
def evaluation(args, model, tokenizer, eval_data, eval_dataloader):
    # f1, kf1, bleu, rouge, ppl
    total_loss = 0.
    model.eval()
    with torch.no_grad():
        cnt = 0
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  args.local_rank not in [-1,0]):
            data['labels'][data['labels']==tokenizer.pad_token_id]=-100 # 굉장히 중요.
            data = {i:j.cuda() for i,j in data.items() if i!='ids'}
            loss = model(**data).loss
            if args.distributed:
                torch.distributed.reduce(loss, 0)
                loss = loss / torch.distributed.get_world_size()
            total_loss+=loss.item()
    total_loss = total_loss / len(eval_dataloader)
    # generation        
    model = model.module if hasattr(model,'module') else model        
    predict_result = []
    total_ids = []
    with torch.no_grad():
        cnt = 0
        for data in tqdm(eval_dataloader, desc = 'gen_evaluate', disable =  args.local_rank not in [-1,0]):
            data = {i:j.cuda() for i,j in data.items() if i not in ['labels','id']}
            outputs = model.generate(
                    **data,
                    pad_token_id = tokenizer.pad_token_id,
                    decoder_start_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.eos_token_id,
                    eos_token_id = tokenizer.eos_token_id,
                    early_stopping = True,
                    do_sample = False,
                    num_beams = 20,
					)
            predicts = tokenizer.batch_decode(outputs, skip_special_tokens = True)
            actual = tokenizer.batch_decode(data['labels'], skip_special_tokens = True)
            bs = data['input_ids'].size(0)
            cnt+=bs
            predict_result.extend(predicts)
            
    total_f1 = []
    total_kf1 = []
    total_bleu1 = []
    total_bleu4 = []
    total_rouge_l = []
    
    for data, predict in zip(eval_data, predict_result):
        predict = post_process(predict)
        response = data['answer'].strip()
        total_f1.append(unigram_f1_score(predict, response, None))
        bleu = sentence_bleu_score(response, predict, None)
        total_bleu1.append(bleu[0])
        total_bleu4.append(bleu[1])
        total_rouge_l.append(sentence_rouge_l(response, predict, None))
        knowledge = data['positive_ctxs'][0]['title']+' '+data['positive_ctxs'][0]['context'] # 첫번째 것의 설정.
        total_kf1.append(unigram_f1_score(predict, knowledge, None))
    return dict(total_f1 = total_f1, total_kf1 = total_kf1, total_bleu1 = total_bleu1, total_bleu4 = total_bleu4, total_rouge_l = total_rouge_l, total_loss = total_loss, cnt=cnt), Predict

def merge_scores(scores):
    if args.distributed:
        cnt = sum([j.item() for j in get_global(args, torch.tensor([scores['cnt']]).cuda())])
        f1 = sum([j.item() for j in get_global(args, torch.tensor([sum(scores['total_f1'])]).cuda())])/cnt
        kf1 = sum([j.item() for j in get_global(args, torch.tensor([sum(scores['total_kf1'])]).cuda())])/cnt
        bleu1 = sum([j.item() for j in get_global(args, torch.tensor([sum(scores['total_bleu1'])]).cuda())])/cnt
        bleu4 = sum([j.item() for j in get_global(args, torch.tensor([sum(scores['total_bleu4'])]).cuda())])/cnt
        rougel = sum([j.item() for j in get_global(args, torch.tensor([sum(scores['total_rouge_l'])]).cuda())])/cnt
        
    else:
        cnt = scores['cnt']
        f1 = sum(scores['total_f1'])/cnt
        kf1 = sum(scores['total_kf1'])/cnt
        bleu1 = sum(scores['total_bleu1'])/cnt
        bleu4 = sum(scores['total_bleu4'])/cnt
        rougel = sum(scores['total_rouge_l'])/cnt
    return dict(f1=np.round(f1,4), kf1=np.round(kf1,4), bleu1=np.round(bleu1,4), bleu4=np.round(bleu4,4), rougel=np.round(rougel,4), cnt=cnt, loss = np.round(total_loss,4))

def train():
    if args.local_rank in [-1,0]:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = False, min_difference=1e-5)
    if args.fp16:
        scaler = GradScaler()
    
    flag_tensor = torch.zeros(1).cuda()
    # train
    ########################################################################################
    global_step = 0
    for epoch in range(1, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            #val_sampler.set_epoch(epoch)
        model.train()
        Loss = 0.
        c = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=args.local_rank not in [-1,0])
        #train
        for data in iter_bar:
            optimizer.zero_grad()
            data['labels'][data['labels']==tokenizer.pad_token_id]=-100 # 굉장히 중요.
            data = {i:j.cuda() for i,j in data.items() if i!='ids'}

            if args.fp16:
                with autocast():
                    loss = model.forward(**data).loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()

            else:
                loss = model.forward(**data).loss # relevance_dist : bs, top_n
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            c+=1
            scheduler.step()
            if args.distributed:
                torch.distributed.reduce(loss, 0)
                loss = loss / torch.distributed.get_world_size()
            Loss+=loss.item()
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}",'total_loss':f'{Loss/c:.5f}'}) # 감소한다는 것을 확인하는 것임.
            if global_step%args.logging_term == 0:
                if args.local_rank in [-1,0]:
                    logger1.info(iter_bar)
                    logger2.info(iter_bar)
            global_step+=1
        
        # epoch 당 기록.
        if args.local_rank in [-1,0]:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        # evaluation
        ###################################################################################################
        if args.eval_epoch!=0 and epoch%args.eval_epoch==0:
            
            scores,_= evaluation(args, model, tokenizer, val_data, val_dataloader)
            scores = merge_scores(scores)
            if args.local_rank == 0:
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
                model_to_save = model.module if hasattr(model,'module') else model
                early_stop.check(model_to_save, total_loss)  
                if early_stop.timetobreak:
                    flag_tensor += 1
            if args.distributed:
                torch.distributed.broadcast(flag_tensor, 0) 
                torch.distributed.barrier()
        ###################################################################################################
        if args.early_stop:    
            if flag_tensor:
                if args.local_rank in [-1,0]:
                    logger1.info('early stop')
                    logger2.info('early stop')
                break
    # 저장시 - gpu 0번 것만 저장 - barrier 필수
    if args.local_rank in [-1,0]:
        torch.save(early_stop.best_model, os.path.join(early_stop.save_dir,'best_model'))
        logger1.info('train_end')
        logger2.info('train end')

if __name__=='__main__':
    args  = get_args()
    seed_everything(42)
    logger1, logger2 = get_log(args)
    if args.local_rank in [-1,0]:
        logger1.info(args)
        logger2.info(args)
        
    # tokenizer, model load
    ########################################################################################
    tokenizer = T5Tokenizer.from_pretrained(args.ptm_path, extra_ids = 0)
    
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens" : [AddedToken(BYTE_TOKENS[t]) for t in BYTE_TOKENS]
        }
    )
    
    config = T5Config.from_pretrained(args.ptm_path)
    t5 = T5ForConditionalGeneration.from_pretrained(args.ptm_path)
    model = FiDT5(config)
    model.load_t5(t5.state_dict())
    ########################################################################################
    # distributed 관련
    if args.distributed:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count()>1
        # 이 프로세스가 어느 gpu에 할당되는지 명시
        torch.cuda.set_device(args.local_rank)
        # 통신을 위한 초기화
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    else:
        model.cuda()
    
    # data
    ########################################################################################
    train_data = load_jsonl(args.train_data)#[:8]
    train_data = make_index(train_data) # id 만들어주기
    train_dataset = FiDDataset(args, train_data, tokenizer, args.is_train)
    train_sampler = DistributedSampler(train_dataset) if args.distributed else RandomSampler(train_dataset) # XXX
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size, sampler = train_sampler, collate_fn = train_dataset._collate_fn)
    
    val_data = load_data(args.val_data, args.local_rank, args.distributed)#[:8]
    val_dataset = FiDDataset(args, val_data, tokenizer, args.is_train)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,batch_size = args.batch_size, sampler = val_sampler, collate_fn = val_dataset._collate_fn)
    ########################################################################################
    
    # optimizer, scheduler
    ########################################################################################
    optimizer_grouped_parameters = make_optimizer_group(args, model)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, args.lr, weight_decay = args.decay)
    scheduler = get_scheduler(args, train_dataloader)
    ########################################################################################
    # train
    ########################################################################################
    train()
    ########################################################################################
