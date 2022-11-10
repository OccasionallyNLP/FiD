# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import time
import re
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,DistributedSampler,RandomSampler,SequentialSampler
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
from train_FiD import evaluation, merge_scores

def get_args():
    # parser
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--test_data', type=str, help = 'test_data 위치')
    parser.add_argument('--output_dir', type=str, help = 'output 위치')
    
    
    # PTM model
    parser.add_argument('--ptm_path', type=str)
    # model
    parser.add_argument('--top_n', type=int, default = 5, help = 'top n의 개수')
    parser.add_argument('--contain_title', type=str2bool, default=True)
    parser.add_argument('--answer_max_length', type=int)
    # specific
    parser.add_argument('--is_train', type=str2bool, default = False ,help = 'mode')
    parser.add_argument('--shuffle', type=str2bool, default = False ,help = 'mode')
    parser.add_argument('--include_history', type=str2bool, help = 'include history')
    parser.add_argument('--just_user', type=str2bool, help = 'include history')
    parser.add_argument('--history_turn', type=int, help = 'how many turn do you want to include in history')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--batch_size', default = 8, type=int)

    # TODO
    ## distributed 관련
    parser.add_argument('--local_rank', type=int, default = -1)
    parser.add_argument('--distributed', type=str2bool, default = False)
    args = parser.parse_args()
    return args

def post_process(args, data, predict):
    output = []
    for i in zip(data, predict):
        output_i = dict(answer = i['answer'], question = i['question'], dialog_no = i['dialog_no'])
        output_i['history']=['['+j['speaker']+']'+j['utterance'] for j in i['history']]
        if args.history_turn:
            output_i['history']=output_i['history'][:args.history_turn]
        output_i['history']=' '.join(output_i['history'])
        if args.is_train:
            positive_ctxs=['['+j['title']+']'+j['context'] for j in i['positive_ctxs'][:args.top_n]] 
            for _,j in enumerate(positive_ctxs):
                output_i['positive_ctxs_%d'%(_+1)]=j
            output_i['include_gtk']=True
        else:
            retrieved_ctxs=['['+j['title']+']'+j['context'] for j in i['retrieved_ctxs'][:args.top_n]]
            for _,j in enumerate(retrieved_ctxs):
                output_i['retrieved_ctxs_%d'%(_+1)]=j
            output_i['include_gtk']=(i['positive_ctxs_ids'][0] in i['retrieved_ctxs_ids'][:args.top_n])
        output.append(output_i)
    output = pd.DataFrame(output)
    return output

if __name__=='__main__':
    args  = get_args()
    print(args)
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)    
    # tokenizer, config, model
    ###########################################################################################
    tokenizer = T5Tokenizer.from_pretrained(args.ptm_path, extra_ids=0)
    config = T5Config.from_pretrained(args.ptm_path)
    model = FiDT5(config)
    model.load_state_dict(torch.load(args.model_path))
    
    # TODO
    # distributed 관련
    # if args.distributed:
    #     assert torch.cuda.is_available()
    #     assert torch.cuda.device_count()>1
    #     # 이 프로세스가 어느 gpu에 할당되는지 명시
    #     torch.cuda.set_device(args.local_rank)
    #     # 통신을 위한 초기화
    #     torch.distributed.init_process_group(backend='gloo', init_method='env://')
    #     model.cuda()
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device = args.local_rank)
    if torch.cuda.is_available():
        model.cuda()
    # data
    ###########################################################################################################################################
    test_data = load_data(args.test_data, args.local_rank, args.distributed)
    test_dataset = FiDDataset(args, test_data, tokenizer, args.is_train, args.shuffle)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size, sampler = test_sampler,collate_fn=test_dataset._collate_fn)
    ###########################################################################################################################################
    # time check
    now = time.time()
    
    # evaluation
    ###########################################################################################################################################
    scores, predict_result = evaluation(args, model, tokenizer, test_data, test_dataloader)
    scores = merge_scores(args,scores)
    ppl = np.exp(scores['loss'])
    ###########################################################################################################################################
    
    with open(os.path.join(args.output_dir, 'result.txt'),'w',encoding='utf-8') as f:
        f.write(f'ppl - {ppl}')
        f.write(f'{scores}')
    print(f'score - {scores} - ppl - {ppl}')
    print(f'processing time - {time.time()-now}')
    
    output = post_process(args, test_data, predict_result)
    output.to_csv(os.path.join(args.output_dir, 'predicted.csv'), encoding='cp949')
    output.to_csv(os.path.join(args.output_dir, 'my_predicted.csv'), encoding='utf-8')    
