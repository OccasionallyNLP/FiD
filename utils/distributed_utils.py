# -*- coding: utf-8 -*-
# distributed utils
import torch
from tqdm import tqdm
import math
from torch import tensor as T

# multi gpu 활용시 loss 모아주기
def get_global_loss(args,local_loss):
    loss_to_gather = [torch.zeros_like(local_loss) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list = loss_to_gather, tensor = local_loss)
    global_loss = []
    for i in range(torch.distributed.get_world_size()):
        if i!=args.local_rank:  
            global_loss.append(loss_to_gather[i].to(local_loss.device))
        else:
            global_loss.append(local_loss)
    return global_loss

# 2023.01.01 추가
def distributed_load_data(data, local_rank, distributed,drop_last = True):
    samples = []
    if distributed:
        world_size = torch.distributed.get_world_size()
        if drop_last:
            data = data[:len(data)//world_size*world_size] # drop last 효과
        else:
            num_samples = math.ceil(len(data)/world_size)
            total_size = num_samples*world_size
            padding_size = total_size - num_samples
            if padding_size <= len(data):
                data += data[:padding_size]
            else:
                data += (data*math.ceil(padding_size/len(data)))[:padding_size] 
        num_samples = math.ceil(len(data)/world_size)
        samples = data[local_rank:local_rank+num_samples]
        return samples
    return data

# excec embedding distributed
def exec_embedding_distributed(args,passage_encoder, context_dataloader):
    output = []
    indices = []
    for data in tqdm(context_dataloader, disable = args.local_rank not in [-1,0]):
        passage_encoder.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                data = {i:j.cuda() for i,j in data.items()}
            passage_embedding = passage_encoder(**data)
            output.extend(passage_embedding.cpu().tolist())
            indices.extend(data['labels'].cpu().tolist())
    output = T(output).cuda()
    indices = T(indices).cuda()
    return output, indices

# global output, indices
def get_global_output_indices(args, local_vector, local_indices):
    vector_to_gather = [torch.zeros_like(local_vector) for _ in range(torch.distributed.get_world_size())]
    indices_to_gather = [torch.zeros_like(local_indices) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list = vector_to_gather, tensor = local_vector)
    torch.distributed.all_gather(tensor_list = indices_to_gather, tensor = local_indices)
    global_vectors = []
    global_indices = []
    for i in range(torch.distributed.get_world_size()):
        if i!=args.local_rank:  
            global_vectors.append(vector_to_gather[i].to(local_vector.device))
            global_indices.append(vector_to_gather[i].to(local_indices.device))
        else:
            global_vectors.append(local_vector)
            global_indices.append(local_indices)
    return global_vectors, global_indices

def get_global(args, thing):
    to_gather = [torch.zeros_like(thing) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensor_list = to_gather, tensor = thing)
    global_thing = []
    for j in range(torch.distributed.get_world_size()):
        if j!=args.local_rank:  
            global_thing.extend(to_gather[j].to(thing.device))
        else:
            global_thing.extend(thing)
    return global_thing
