import math 
import numpy as np

import torch
from torch.autograd import Variable



def get_inv(my_map):
    inv_map = {v: k for k, v in my_map.items()}
    return inv_map

def get_set(all,perm):
    new_set=list()
    for i in range(len(all)):
        if i in perm:
            new_set.append(all[i])
    return new_set

def Var(v,CUDA):
    if CUDA:
        return Variable(v.cuda())
    else:
        return Variable(v)

def id_to_tag(seq,to_tag):
    res=list()

    for ele in seq:
        res.append(to_tag[ele])

    return res

def convert_2_tensor(seq, to_ix, dt, CUDA):
    if to_ix == None:
        return Var(torch.tensor(seq, dtype=dt))
    else:
        idxs = list()
        for w in seq:
            if w in to_ix:
                idxs.append(to_ix[w])
        return Var(torch.tensor(idxs, dtype=dt),CUDA)

def convert_2_class_tensor(target, to_ix, dt, CUDA):
    if to_ix == None:
        print("error occured in convert_2_class_tensor")
    else:
        idxs = [0 for x in to_ix]
        idxs[to_ix[target]] = 1
        return Var(torch.tensor([idxs], dtype=dt),CUDA)


def predict_class(class_score, CUDA):
    if CUDA:
        class_score = class_score.cpu().detach().numpy()
    else:
        class_score = class_score.detach().numpy()
    classes = list()
    for seq in class_score:
        classes.append(np.argmax(seq))
    return classes



def getFolds(fold, data, div = 8):
    """
    Splits the given data into a different fold based on input
    """
    if fold < 1  or fold > div:
        print("Incorrect div to fold number encountered")
        return None, None
    testSize = math.floor(len(data)/div)
    beforeTest = data[:(fold-1)*testSize]
    test = data[(fold-1)*testSize:fold*testSize]
    afterTest = data[fold*testSize:]
    return test, beforeTest + afterTest

    
def getSizeData(data):
    """
    Description: counts the number of individual annotations of each type in the data.
    """
    last = ""
    freq = {}
    for x in data:
        for y in x.target:
            if y == last: 
                continue
            else:
                last = y
                try:
                    freq[y] += 1
                except KeyError:
                    freq[y] = 1
    try:
        del freq["o"]
    except KeyError:
        print(data)
    return freq


def getBatches(data,batchSize=64):
    return [data[i*batchSize:(i+1)*batchSize] for i in range(math.ceil(len(data)/batchSize))]


def expand_annotations(tokens,target,tokenizer):
    i=-1
    new_annotations=list()
    for token in tokens:
        i=i+1

        tokenized=tokenizer.tokenize(token)

        if len(tokenized)==1:
            new_annotations.append(target[i])
        else:
            new_annotations=new_annotations+len(tokenized)*[target[i]]

    return new_annotations



def expand_tokens(tokens,tokenizer):
    new_tokens=list()
    for token in tokens:
        new_tokens=new_tokens+tokenizer.tokenize(token)
    return new_tokens



def contract_annotations(tokens,target,tokenizer):

    new_annotations=list()

    for token in tokens:
        cur_list=list()
        tokenized=tokenizer.tokenize(token)

        for i in range(len(tokenized)):
            cur_list.append(target[0])
            target.pop(0)
        new_annotations.append(cur_list)
    return new_annotations


def merge_annotations(annotations):

    new_annotations=list()

    for li in annotations:
        if len(li)==1:
            new_annotations.append(li[0])
        elif len(li) == 0:
            new_annotations.append('o')
        else:
            most_frq=max(set(li), key = li.count)
            new_annotations.append(most_frq)


    return new_annotations