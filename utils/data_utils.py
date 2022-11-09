# data_utils
# -*- coding: utf-8 -*-
# data utils 로 빼두기.
import json
import os
import hashlib
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import tensor as T
from typing import List
import random

class FiDDataset(Dataset):
    # knowledge
    title_prefix = 'UNUSED0000'
    context_prefix = 'UNUSED0001'
    # speaker
    apperentice_prefix = 'UNUSED0002'
    wizard_prefix = 'UNUSED0003'
    # history
    history_prefix = 'UNUSED0004'
    # question
    question_prefix = 'UNUSED0005'
    
    def __init__(self, args, data:List[dict], tokenizer, is_train=True, shuffle=False):
        super().__init__()
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        # is_train -> 원 논문에서는 fusion in decoder의 input으로는 검색 결과가 달려있는 input을 활용함
        # 정답이 있는 지 없는 지에 대한 부분은 없음. (즉, 검색 결과가 틀렸을 경우도 존재할 수 있음)
        # inductive bias로 정답 지식을 default로 주느냐 마느냐에 대한 config 값임.
        self.is_train = is_train # train이 아닐 경우엔, positive_ctxs_ids가 없음
        self.shuffle = shuffle # is_train mode 일 때, 첫번째가 항상 정답 지식인 것을 모델이 학습하는 것을 방지하기 위해서 활용함.
        
    def get_feature(self,index):
        ```
        input - knowledge, history, question
        ```
        # question
        question = self.question_prefix+self.data[index]['question']
        if self.is_train:
            gold_knowledge =  [self.title_prefix+s['title']+self.context_prefix+s['context'] if self.args.contain_title else self.context_prefix+s['context'] for s in self.data[index]['positive_ctxs']]
            if self.args.top_n == 1:
                total_knowledge = [gold_knowledge[0]]
            else:
                ## 복수 가능
                indice = []
                candidate_ctxs_ids = [i for i in self.data[index]['retrieved_ctxs_ids']]
                candidate_knowledge = [self.title_prefix+i['title']+self.context_prefix+i['context'] if self.args.contain_title else self.context_prefix+i['context'] for i in self.data[index]['retrieved_ctxs']]
                for i in self.data[index]['positive_ctxs_ids']:
                    if i in candidate_ctxs_ids:
                        idx = candidate_ctxs_ids.index(i)
                        indice.append(idx)
                candidate_ctxs_ids = [i for i in candidate_ctxs_ids if i not in indice]
                candidate_ctxs = [i for _,i in enumerate(candidate_ctxs) if _ not in indice]
                candidate_knowledge = [i for i in candidate_ctxs[:max(0,self.args.top_n-len(gold_knowledge))]]
                total_knowledge = gold_knowledge + candidate_knowledge
                assert len(total_knowledge)>=self.args.top_n
                # 정답 지식이 항상 첫번째에 있는 것을 모델이 학습하는 것을 방지하기 위함임.
                if self.shuffle:
                    total_knowledge = random.shuffle(total_knowledge)
        else:
            candidate_knowledge = [self.title_prefix+i['title']+self.context_prefix+i['context'] if self.args.contain_title else self.context_prefix+i['context'] for i in self.data[index]['retrieved_ctxs']]
            assert len(candidate_knowledge)>=self.args.top_n
            total_knowledge = candidate_knowledge[:self.args.top_n]
            
        # history 
        if self.args.include_history:
            history = []
            for i in self.data[index]['history']:
                if i['speaker'] == 'agent':
                    if self.args.just_user:
                        continue
                    history.append(self.wizard_prefix+i['utterance'])
                else:
                    history.append(self.apperentice_prefix+i['utterance'])
            checked_history = self.check_history(history, self.args.history_turn)
            checked_history = self.history_prefix+''.join(checked_history)
        # history가 없다면
        else:
            checked_history = ''
        
        total = [i+checked_history+question for i in total_knowledge] 
        
        # answer
        answer = self.data[index]['answer']
        
        # id
        # for distributed
        if self.data[index].get('id',-1)!=-1:
            ids = T(self.data[index]['id'])
        else:
            ids = None
        #return output
        return ids, total, answer
        
    # history 길이 제한 - 없앰 - turn 수로 구분
    # 20220506 history - question 분리
    def check_history(self, history, how_many=None):
        checked_history = history
        checked_history.reverse()
        if how_many is not None:
            assert how_many>=1
            checked_history = checked_history[:how_many]
        checked_history.reverse()
        return checked_history
    
    def _collate_fn(self, batch):
        encoder_inputs = []
        labels = []
        indice = []
        for (ids,total,answer) in batch:
            encoder_inputs.extend(total)
            labels.append(answer)
            indice.append(ids)
        labels = self.tokenizer(labels,padding='longest',return_tensors = 'pt').input_ids
        bs = labels.size(0)
        encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
        input_ids, attention_mask = encoder_inputs.input_ids.reshape(bs,self.args.top_n,-1), encoder_inputs.attention_mask.reshape(bs,self.args.top_n,-1)
        return dict(input_ids = input_ids, attention_mask = attention_mask, labels = labels, ids=T(indice))
    
    def __getitem__(self, index):
        return self.get_feature(index)
    
    def __len__(self):
        return len(self.data)

class T5Dataset(Dataset):
    # speaker
    apperentice_prefix = 'UNUSED0002'
    wizard_prefix = 'UNUSED0003'
    # history
    history_prefix = 'UNUSED0004'
    # question
    question_prefix = 'UNUSED0005'
    
    def __init__(self, args, data:List[dict], tokenizer):
        super().__init__()
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        
    def get_feature(self,index):
        ```
        input - knowledge, history, question
        ```
        # question
        question = self.question_prefix+self.data[index]['question']
        
        # history 
        if self.args.include_history:
            history = []
            for i in self.data[index]['history']:
                if i['speaker'] == 'agent':
                    if self.args.just_user:
                        continue
                    history.append(self.wizard_prefix+i['utterance'])
                else:
                    history.append(self.apperentice_prefix+i['utterance'])
            checked_history = self.check_history(history, self.args.history_turn)
            checked_history = self.history_prefix+''.join(checked_history)
        # history가 없다면
        else:
            checked_history = ''
        
        total = checked_history+question
        
        # answer
        answer = self.data[index]['answer']
        
        # id
        # for distributed
        if self.data[index].get('id',-1)!=-1:
            ids = T(self.data[index]['id'])
        else:
            ids = None
        #return output
        return ids, total, answer
        
    def check_history(self, history, how_many=None):
        checked_history = history
        checked_history.reverse()
        if how_many is not None:
            assert how_many>=1
            checked_history = checked_history[:how_many]
        checked_history.reverse()
        return checked_history
    
    def _collate_fn(self, batch):
        encoder_inputs = []
        labels = []
        indice = []
        for (ids,total,answer) in batch:
            encoder_inputs.append(total)
            labels.append(answer)
            indice.append(ids)
        labels = self.tokenizer(labels,padding='longest',return_tensors = 'pt').input_ids
        encoder_inputs = self.tokenizer(encoder_inputs, padding='longest',return_tensors = 'pt')
        return dict(input_ids = encoder_inputs.input_ids, attention_mask = encoder_inputs.attention_mask, labels = labels, ids=T(indice))
    
    def __getitem__(self, index):
        return self.get_feature(index)
    
    def __len__(self):
        return len(self.data)
