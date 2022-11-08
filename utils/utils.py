# utils
# -*- coding: utf-8 -*-
import json
import os
import hashlib
from tqdm import tqdm
import numpy as np
import torch
from typing import List
import random
import argparse
import logging
import copy

# jsonl save, load
def save_jsonl(address,data,name):
    f = open(os.path.join(address,name+'.jsonl'),'w',encoding = 'utf-8')
    for i in data:
        f.write(json.dumps(i,ensure_ascii=False)+'\n') # for korean
        
def load_jsonl(path):
    result = []
    f = open(path,'r',encoding = 'utf-8')
    for i in tqdm(f):
        result.append(json.loads(i))
    return result 

def make_index(data):
    for _,i in enumerate(data):
        if i.get('id',-1)!=-1:
            break
        else:
            i['id']=_
    return data       

def load_data(data_path, local_rank, distributed):
    data = load_data(data_path)
    data = make_index(data)
    samples = []
    if distributed:
        world_size = torch.distributed.get_world_size()
        for k,example in enumerate(data):
            if not k%world_size == local_rank:
                continue
            samples.append(example)
        return samples
    return data
    
# compute hash
def compute_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# bool for argparse
def str2bool(v):
    """
    Transform user input(argument) to be boolean expression.
    :param v: (string) user input
    :return: Bool(True, False)
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# seed
def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_log(args):
    #global logger1, logger2
    logger1 = logging.getLogger('train_file') # 적지 않으면 root로 생성
    logger2 = logging.getLogger('stream') # 적지 않으면 root로 생성
    
    # 2. logging level 지정 - 기본 level Warning
    logger1.setLevel(logging.INFO)
    logger2.setLevel(logging.INFO)
    # 3. logging formatting 설정 - 문자열 format과 유사 - 시간, logging 이름, level - messages
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] >> %(message)s')

    # 4. handler : log message를 지정된 대상으로 전달하는 역할.
    # SteamHandler : steam(terminal 같은 console 창)에 log message를 보냄
    # FileHandler : 특정 file에 log message를 보내 저장시킴.
    # handler 정의
    stream_handler = logging.StreamHandler()
    # handler에 format 지정
    stream_handler.setFormatter(formatter)
    # logger instance에 handler 삽입
    logger2.addHandler(stream_handler)
    os.makedirs(args.output_dir,exist_ok=True)
    if args.test_name is None:
      args.test_name = 'log'
    file_handler = logging.FileHandler(os.path.join(args.output_dir,'train_%s.txt'%(args.test_name)), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger1.addHandler(file_handler)
    file_handler2 = logging.FileHandler(os.path.join(args.output_dir,'train_val_%s.txt'%(args.test_name)), encoding='utf-8')
    file_handler2.setFormatter(formatter)
    return logger1, logger2

# early stop
class EarlyStopping(object):
    def __init__(self, patience, save_dir, max = True, min_difference=1e-5):
        self.patience = patience
        self.min_difference = min_difference
        self.max = max
        self.score = -float('inf') if max else float('inf')
        self.best_model = None
        self.best_count = 0
        self.timetobreak = False
        self.save_dir = save_dir
    
    def check(self, model, calc_score):
        if self.max:
            if self.score-calc_score<self.min_difference:
                self.score = calc_score
                self.best_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
            else:
                self.best_count+=1
                if self.best_count>=self.patience:
                    self.timetobreak=True
                    torch.save(self.best_model, os.path.join(self.save_dir,'best_model'))
        else:
            if self.score-calc_score>self.min_difference:
                self.score = calc_score
                self.best_count = 0
                self.best_model = copy.deepcopy(model.state_dict())
            else:
                self.best_count+=1
                if self.best_count>=self.patience:
                    self.timetobreak=True
                    torch.save(self.best_model, os.path.join(self.save_dir,'best_model'))
