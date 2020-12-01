# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.autograd import Function


def predict_class(class_score,CUDA):
    if CUDA:
        class_score=class_score.cpu().detach().numpy()
    else:
        class_score = class_score.detach().numpy()
    classes=list()
    for seq in class_score:
        classes.append(np.argmax(seq))
    return classes



class Classification(torch.nn.Module):
    def __init__(self, num_class,hidden_dim):
        super(Classification, self).__init__()

        self.num_class=num_class
        self.label = nn.Linear(hidden_dim, num_class)

    def forward(self,input):

        outp=self.label(input)
        class_score = F.log_softmax(outp.view(-1, self.num_class), dim=1)
        return class_score






class Embedding(nn.Module):
    def __init__(self, emb_dim,vocab_size,initialize_emb,word_to_ix):
        super(Embedding, self).__init__()


        self.embedding=nn.Embedding(vocab_size,emb_dim)

        if initialize_emb:
            inv_dic = {v: k for k, v in word_to_ix.items()}

            for key in initialize_emb.keys():
                if key in word_to_ix:
                    ind = word_to_ix[key]
                    self.embedding.weight.data[ind].copy_(torch.from_numpy(initialize_emb[key]))




    def forward(self,input):

        return self.embedding(input)




class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, tagset_size,bidirec):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim= input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim,bidirectional=bidirec)
        if bidirec:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim , tagset_size)

    def forward(self, input):
        lstm_tuple = self.lstm(input.view(len(input), 1, -1))
        tag_space = self.hidden2tag(lstm_tuple[0].view(len(lstm_tuple[0]), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return (tag_scores,lstm_tuple)







