##############
# 전처리
import os
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from collections import Counter
from typing import Dict, List
import copy
from pprint import pprint

def save_jsonl(address,data,name):
    f = open(os.path.join(address,name+'.jsonl'),'w',encoding = 'utf-8')
    for i in data:
        f.write(json.dumps(i,ensure_ascii=False)+'\n')

def load_jsonl(path):
    result = []
    f = open(path,'r',encoding = 'utf-8')
    for i in tqdm(f):
        result.append(json.loads(i))
    return result 

# data, knowledge
total_data = json.load(open('TEXTNET_dataset_(4183set)_1109_json.json',encoding='utf-8-sig'))
knowledge = json.load(open('./지식정보구조221021.json',encoding='utf-8-sig'))

def make_knowledge(knowledge):
    result = {}
    for i in tqdm(knowledge):
        result[i['knowledge_id']]=dict(title=i['attribute'], context=i['text'], knowledge_topic=i['knowledge_topic'])
    # knowledge -> NONE tag
    result['NONE']=dict(title='',context='')
    return result

knowledge = make_knowledge(knowledge['knowledge_list'])
db_id_to_knowledge_id = {j:i for i,j in enumerate(knowledge.keys())}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('../../../models/kt-ulm-base')

check_len = []
for i in knowledge.values():
    check_len.append(len(tokenizer.tokenize(i['title']+' '+i['context'])))

print(np.percentile(check_len, [0,50,95,99,100]))

# CHECK 지식 기반 대화 주제

knowledge_data = load_jsonl('/home/work/nlp/user/ok/data/additional_data/221027_second/4/knowledge_train_data.jsonl')

check_knowledge = []
for i in knowledge_data:
    check_knowledge.extend(i['positive_ctxs_ids'])
topics = []
for i in check_knowledge:
    topics.append(knowledge[kid_db_id[i]]['knowledge_topic'])

from collections import Counter,OrderedDict

check_knowledge = Counter(topics).most_common()

for i,j in check_knowledge:
    print(f'{i} : {j}')

dic = {}
for i in check_knowledge.most_common():
    dic[knowledge[kid_db_id[i[0]]]['knowledge_topic']]=i[1]

sorted_dict = sorted(dic.items(), key = lambda item: item[0], reverse = True)
print(sorted_dict)

kid_db_id = {j:i for i,j in db_id_to_knowledge_id.items()}

check_knowledge[

# DKLEE가 준 data 활용  : 221103
## 지식 없음은 None tag

# dklee가 주신 data
attached_data = load_jsonl('./dpr_whole_221101_add_retrieval.jsonl')

for i in attached_data:
    if i['dialog_no']==3:
        if i['utterance_no']==20:
            print(i)
            break

tag_attach = {}
for i in tqdm(attached_data):
    if tag_attach.get(i['dialog_no']):
        tag_attach[i['dialog_no']][i['utterance_no']] = i['retrieval_knowledge_ids']
    else:
        tag_attach[i['dialog_no']] = {i['utterance_no'] : i['retrieval_knowledge_ids']}
for i,j in tag_attach.items():
    for k in j.values():
        assert len(k) == 100

# 10.27 data, knowledge
total_data = json.load(open('./221027/221027.json',encoding='utf-8-sig'))
knowledge = json.load(open('./221027/지식정보구조221027.json',encoding='utf-8-sig'))
knowledge = make_knowledge(knowledge['knowledge_list'])
db_id_to_knowledge_id = {j:i for i,j in enumerate(knowledge.keys())}

## data consistency

total_data_to_dict = {}
for i in total_data['dialog_list']:
    #total_data_to_dict[i['dialog_no']]={}
    dic = {}
    for j in i['dialog_contents']:
        dic[int(j['utterance_no'])]=j['used_knowledge']
    total_data_to_dict[i['dialog_no']]=dic

for k in [5,20,100]:
    check = []
    for dialog_no in tag_attach:
        for utterance_no, value in tag_attach[dialog_no].items():
            p = total_data_to_dict[dialog_no][utterance_no]
            if p:
                p = p[0]
            else:
                p = 'NONE'
            check.append(p in value[:k])
    print(f'hit - {k} ---- {sum(check)/len(check)}') 

train_data = load_jsonl('./221027/train_data.jsonl')
val_data = load_jsonl('./221027/val_data.jsonl')
test_data = load_jsonl('./221027/test_data.jsonl')

# tag_attach
check = []
check_consistency = []
for data in [train_data, val_data, test_data]:
    for i in data:
        for j in i:
            if j['positive_ctxs_ids']:
                try:
                    j['retrieved_ctxs_ids']=[db_id_to_knowledge_id[k] for k in tag_attach[j['dialog_no']][j['utterance_no']]]
                    chk = [j['positive_ctxs_ids'][0] in j['retrieved_ctxs_ids'][:k] for k in [5,20,100]]
                    check_consistency.append(chk)
                    
                except:
                    check.append(dict(dialog_no=j['dialog_no'], utterance_no=j['utterance_no']))
                    del j

print(sum(list(map(lambda i : i[0], check_consistency)))/len(check_consistency))
print(sum(list(map(lambda i : i[1], check_consistency)))/len(check_consistency))
print(sum(list(map(lambda i : i[2], check_consistency)))/len(check_consistency))

before_data = load_jsonl('/home/work/nlp/user/ok/data/additional_data/221027_second/second_version/attached/4/knowledge_train_data.jsonl')

for k in [5,20,100]:
    check = []
    for utterance_i in before_data:
        check.append(utterance_i['positive_ctxs_ids'][0] in utterance_i['retrieved_ctxs_ids'][:k])
        
    print(f'hit - {k} ---- {sum(check)/len(check)}') 

# data의 version이 2개이다

version 0 - original  
version 1 - additional question(6개가 추가된 데이터)

def split_data_by_version(data_dialog_list):
    version_0 = []
    version_1 = []
    for _,i in enumerate(data_dialog_list):
        for j in i['dialog_contents']:
            if '-' in j['utterance_no']:
                version_1.append(_)
                break
        else:
            version_0.append(_)
    data_v0 = [data_dialog_list[i] for i in version_0]
    data_v1 = [data_dialog_list[i] for i in version_1]
    return data_v0, data_v1

data_v0, data_v1 = split_data_by_version(total_data['dialog_list'])

print(f'version 0 - {len(data_v0)}')
print(f'version 1 - {len(data_v1)}')

# 통합 데이터

def data_organize(db_id_to_knowledge_id:Dict, knowledges:Dict, data:List):
    new_data = []
    session_cnt = 0
    cnt = 0
    knowledge_cnt = 0
    check_cnt = 0
    for _,i in enumerate(data):
        session = []
        session_cnt+=1
        for _,j in enumerate(i['dialog_contents']):
            # 필요 사항 - dialog no, utterance no, dialog act, 지식
            utt = dict(positive_ctxs_ids=[db_id_to_knowledge_id[k.upper()] for k in j['used_knowledge']],
                       positive_ctxs = [knowledges[k.upper()] for k in j['used_knowledge']],
                       dialog_act = j['dialog_act'],
                       speaker=j['speaker'].lower(), utterance=j['utterance'], utterance_no=j['utterance_no'], dialog_no=i['dialog_no'])
            cnt+=1
            if j['speaker'].lower() == 'agent':
                check_cnt+=1
                if j['used_knowledge']:
                    knowledge_cnt+=1
            session.append(utt)
        new_data.append(session)
    print(f'# of session : {session_cnt}')
    print(f'# of utterance : {cnt}')
    print(f'# of utterance of agent : {check_cnt}')
    print(f'# of utterance of using knowledge : {knowledge_cnt}')
    return new_data

data_2_v0 = data_organize(db_id_to_knowledge_id, knowledge, data_v0)
print()
data_2_v1 = data_organize(db_id_to_knowledge_id, knowledge, data_v1)

class DataOrganize(object):
    def __init__(self, data, db_id_to_knowledge_id, knowledge, verbose=False):
        self.data = copy.deepcopy(data)
        self.db_id_to_knowledge_id = db_id_to_knowledge_id
        self.knowledge_id_to_db_id = {i:j for j,i in db_id_to_knowledge_id.items()}
        self.knowledge = copy.deepcopy(knowledge)
        self.splitted_data = None
        self.merged_data = None
        self.verbose = verbose
        self.empathy_data = None # 공감
        self.situation_navigation_data = None # situation navigation
        self.knowledge_data = None # 지식 기반 대화
        self.summarization_data = None # 요약
    
    def make_history_question(self, history):
        # 
        copy_history = copy.deepcopy(history)
        chk = None
        for _,i in enumerate(history[::-1]):
            if i['speaker']=='user':
                chk = _
                break
        # 질문이 없는 경우 제외.
        assert chk is not None
        question = history[-(chk+1)]
        copy_history.pop(-(chk+1))
        return question, copy_history
    
    def split_session(self):
        # 탐침, 정보제공 단계를 나눔.
        output = []
        for session_i in tqdm(self.data):
            for _,i in enumerate(session_i):
                if i['positive_ctxs_ids']:
                    if session_i[_-1]['dialog_act']=='공감': # 탐침의 마지막 단계가 공감이 아닌 경우
                        output.append((session_i[:_],session_i[_:]))
                    else:
                        # 지식 기반 대화 이전 발화의 dialog act가 공감이 아닌 경우 오류임.
                        if self.verbose:
                            print('오류임')
                            print(session_i[_-1]['dialog_no'],session_i[_-1]['utterance_no'])
                    break    
        self.splitted_data = output
    
    def merge_utterance(self):
        # 지식 부분만 merge
        if self.splitted_data is None:
            self.split_session()
        output = []
        check = 0
        for session_i in tqdm(self.splitted_data):
            probing_i, information_i = session_i
            dialog_no = probing_i[0]['dialog_no']
            # 꼴 맞추기
            dialog_act_i = []
            speaker = None
            utterance_i = []
            positive_ctxs_ids = []
            retrieved_ctxs_ids = []
            output_i = []
            for i in information_i:
                if not i['utterance'].startswith('<정보') and not i['utterance'].startswith('<답변'):
                    if not speaker:
                        speaker = i['speaker']
                        utterance_i.append(i['utterance'])
                        positive_ctxs_ids.extend(i['positive_ctxs_ids'])
                        dialog_act_i.append(i['dialog_act'])
                        if i.get('retrieved_ctxs_ids'):
                            retrieved_ctxs_ids.extend(i['retrieved_ctxs_ids'])
                    else:
                        if i['speaker']==speaker:
                            utterance_i.append(i['utterance'])
                            positive_ctxs_ids.extend(i['positive_ctxs_ids'])
                            dialog_act_i.append(i['dialog_act'])
                            if i.get('retrieved_ctxs_ids'):
                                retrieved_ctxs_ids.extend(i['retrieved_ctxs_ids'])
                        # 아니면
                        else:
                            # positive_ctxs_ids가 2개 이상인 경우 제외
                            if len(list(set(positive_ctxs_ids)))<2:
                                #dialog_act_i = list(set(dialog_act_i))
                                if len(retrieved_ctxs_ids)>100:
                                    # 연속 발화인 경우임 -> 태깅 오류
                                    check+=1
                                    continue
                                output_i.append(dict(speaker=speaker, utterance = ' '.join(utterance_i), positive_ctxs_ids = list(set(positive_ctxs_ids)), dialog_act = dialog_act_i[0],\
                                                     retrieved_ctxs_ids = retrieved_ctxs_ids , dialog_no = dialog_no))
                            # reset
                            speaker = i['speaker']
                            utterance_i = [i['utterance']]
                            positive_ctxs_ids = i['positive_ctxs_ids']
                            dialog_act_i = [i['dialog_act']]
                            if i.get('retrieved_ctxs_ids'):
                                retrieved_ctxs_ids=i['retrieved_ctxs_ids']
                else:
                    break
            # 마지막엔 해줘야한다.
            # positive_ctxs_ids가 2개 이상인 경우 제외
            if len(list(set(positive_ctxs_ids)))<2:
                if len(retrieved_ctxs_ids)>100:
                    # 연속 발화인 경우임 -> 태깅 오류
                    check+=1
                    continue
                output_i.append(dict(speaker=speaker, utterance = ' '.join(utterance_i), positive_ctxs_ids = list(set(positive_ctxs_ids)), dialog_act = dialog_act_i[0],\
                                     retrieved_ctxs_ids = retrieved_ctxs_ids, dialog_no = dialog_no))
                
            output.append((probing_i,output_i))
            self.merged_data = output
        print(check)
    def attach_history(self):
        if self.merged_data is None:
            self.merge_utterance()
        output = []
        E = []
        SN = []
        K = []
        S = []
        for session_i in tqdm(self.merged_data):
            probing_i, information_i = session_i
            # 탐침 단계
            for _,i in enumerate(probing_i):
                if _>0: 
                    if i['speaker']=='agent':
                        dialog_act = i['dialog_act'] # 공감, 나머지
                        positive_ctxs=[self.knowledge[self.knowledge_id_to_db_id[j]] for j in i['positive_ctxs_ids']] if i['positive_ctxs_ids'] else [dict(title='',context='')]
                        # 공감 일 때 (공감, 요약)
                        if dialog_act == '공감':
                            history = [dict(speaker=j['speaker'], utterance=j['utterance'], dialog_act=j['dialog_act']) for j in probing_i[:_]]
                            question, history = self.make_history_question(history)
                            output_i = dict(answer=i['utterance'], positive_ctxs=positive_ctxs, positive_ctxs_ids = i['positive_ctxs_ids'], question=question['utterance'],\
                                            history=copy.deepcopy(history), dialog_no = i['dialog_no'])
                            # 요약
                            if _ == len(probing_i)-1:
                                # print('요약')
                                # print(history)
                                # print(question)
                                # print(i['utterance'])
                                # input()
                                S.append(output_i)
                            # 공감
                            else:
                                # print('공감')
                                # print(history)
                                # print(question)
                                # print(i['utterance'])
                                # input()
                                E.append(output_i)
                            
                        # 상황 탐색 일 떄
                        else:
                            history = [dict(speaker=j['speaker'], utterance=j['utterance'], dialog_act=j['dialog_act']) for j in probing_i[:_] if j['dialog_act']!='공감']
                            question, history = self.make_history_question(history)
                            # print('상황탐색')
                            # print(history)
                            # print(question)
                            # print(i['utterance'])
                            # input()
                            output_i = dict(answer=i['utterance'], positive_ctxs=positive_ctxs, positive_ctxs_ids = i['positive_ctxs_ids'], question=question['utterance'],\
                                            history=copy.deepcopy(history), dialog_no = i['dialog_no'])
                            SN.append(output_i)
                else:
                    # 질문이 없는 경우 제외. 이 부분은 자동 생성 해줄 것임.
                    pass
            # knowledge는 아직 미정
            # probing_history = [dict(speaker=j['speaker'], utterance=j['utterance'], dialog_act=j['dialog_act']) for j in probing_i if j['dialog_act']!='공감']
            # for _,i in enumerate(information_i):
            #     if i['speaker']=='agent':
            #         history = probing_history + [dict(speaker=j['speaker'], utterance=j['utterance'], dialog_act=j['dialog_act']) for j in information_i[:_] if j['dialog_act']!='공감']
            #         question, history = self.make_history_question(history)
            #         positive_ctxs=[self.knowledge[self.knowledge_id_to_db_id[j]] for j in i['positive_ctxs_ids']] if i['positive_ctxs_ids'] else [dict(title='',context='')]
            #         retrieved_ctxs=[self.knowledge[self.knowledge_id_to_db_id[j]] for j in i['retrieved_ctxs_ids']] if i['retrieved_ctxs_ids'] else []
            #         output_i = dict(answer=i['utterance'], positive_ctxs=positive_ctxs, positive_ctxs_ids = i['positive_ctxs_ids'], retrieved_ctxs_ids = i['retrieved_ctxs_ids'], retrieved_ctxs=retrieved_ctxs, question=question['utterance'], history=copy.deepcopy(history),  dialog_no = i['dialog_no'])
            #         K.append(output_i)
                    
        self.empathy_data = E
        self.situation_navigation_data = SN
        self.knowledge_data = K
        self.summarization_data = S
    
    def post_process(self, summarization = True):
        # NONE TAG 붙여주고, retrieved ctxs 없어도 붙여주기.
        data = [self.empathy_data, self.situation_navigation_data, self.knowledge_data]
        if summarization:
            data.append(self.summarization_data)
        for i in data:
            for j in i:
                if not j['positive_ctxs_ids']:
                    j['positive_ctxs_ids']=[self.db_id_to_knowledge_id['NONE']]
                    j['positive_ctxs']=[dict(title='',context='')]
        # check = None
        # for i in data:
        #     for j in i:
        #         if check is None:
        #             check = sorted(list(j.keys()))
        #         else:
        #             assert check == sorted(list(j.keys()))

data_2 = data_2_v0+data_2_v1
DO = DataOrganize(data_2, db_id_to_knowledge_id, knowledge, verbose=True)
DO.attach_history()
DO.post_process()
print(f'공감대화 {len(DO.empathy_data)}')
print(f'상황 탐색 {len(DO.situation_navigation_data)}')
print(f'지식 기반 대화 {len(DO.knowledge_data)}')
print(f'상황 요약 {len(DO.summarization_data)}')

# 기존의 val, test data 가져오기

val_data = load_jsonl('../221027/val_data.jsonl')
test_data = load_jsonl('../221027/test_data.jsonl')

val_dialog_no = []
for i in val_data:
    val_dialog_no.append(i[0]['dialog_no'])
test_dialog_no = []
for i in test_data:
    test_dialog_no.append(i[0]['dialog_no'])

# 기존 val, test session 살리기

val_data = []
test_data = []
train_data = []
for i in data_2:
    if i[0]['dialog_no'] in val_dialog_no:
        val_data.append(i)
    elif i[0]['dialog_no'] in test_dialog_no:
        test_data.append(i)
    else:
        train_data.append(i)

for i,j in zip([train_data, val_data, test_data],['train_data','val_data','test_data']):
    TDT = DataOrganize(i, db_id_to_knowledge_id, knowledge)
    TDT.attach_history()
    TDT.post_process(True)
    print(f'공감 대화 {len(TDT.empathy_data)}')
    print(f'상황 탐색 {len(TDT.situation_navigation_data)}')
    print(f'지식 기반 대화 {len(TDT.knowledge_data)}')
    print(f'공감 요약 대화 {len(TDT.summarization_data)}')
    save_jsonl('./3',TDT.empathy_data+TDT.summarization_data,'e_'+j)
    save_jsonl('./3',TDT.situation_navigation_data,'sn_'+j)
    save_jsonl('./3',TDT.knowledge_data,'k_'+j)    
    
    save_jsonl('./4',TDT.empathy_data,'e_'+j)
    save_jsonl('./4',TDT.summarization_data,'s_'+j)
    save_jsonl('./4',TDT.situation_navigation_data,'sn_'+j)
    save_jsonl('./4',TDT.knowledge_data,'k_'+j)    

# 1차 데이터 병합

sn_train_data = load_jsonl('./4/sn_train_data.jsonl')
sn_train_data_1 = load_jsonl('../V1/sn_data.jsonl')
sn_train_data+=sn_train_data_1
save_jsonl('./4',sn_train_data,'sn_train_data_w_v1')
