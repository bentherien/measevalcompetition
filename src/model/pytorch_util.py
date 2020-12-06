# _*_ coding: utf-8 _*_
import os
import numpy as np

from helpers import *
from allennlp.modules.elmo import Elmo,batch_to_ids

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
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

class Sequence_Tagger(nn.Module):
    def __init__(self,word_to_idx,tag_to_ix,pos_to_ix,pretrained_emb,emb_dim,d_model,emb_type,sys_path):
        super().__init__()

        #self.emb=Embedding(emb_dim,len(word_to_idx),pretrained_emb,word_to_idx)
        options_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_options.json")
        weight_file = ""
        
        if emb_type == "original":
            self.emb = "original"
            weight_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        else:
            self.emb = "pubmed"
            weight_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")

        
        self.word_to_idx = word_to_idx
        self.tag_to_ix = tag_to_ix
        self.pos_to_ix = pos_to_ix

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)
        self.pos_tagger = LSTMTagger(emb_dim,d_model//2,len(pos_to_ix),True)
        self.tagger = LSTMTagger(emb_dim,d_model,len(tag_to_ix),True)


    def forward(self, input, CUDA):

        x = self.elmo(Var(batch_to_ids([input]),CUDA))
        x = x["elmo_representations"][0]
        #print("size x,x.squeeze:",x.size(),x.squeeze().size())
        pos_predictions,lstm1_out = self.pos_tagger(x.squeeze())
        predictions, lstm_out = self.tagger(lstm1_out[0].squeeze())

        return predictions, pos_predictions
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        self.load_state_dict(torch.load(filepath))


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for tokenList in testData:

                input = tokenList

                output_task, output_pos = self(input, CUDA)
                pred = predict_class(output_task, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)

        return all_predictions


class GCNNLayer(nn.Module):
    def __init__(self,emb_dim,d_model):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.deps = ["subtok","ROOT", "acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "csubj", "csubjpass", 'dative', "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nmod", "npadvmod", "nsubj", "nsubjpass", "nummod", "oprd", "parataxis", "pcomp", "pobj", "poss", "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp"]
        self.deps += ["r-" + x for x in self.deps]
        MAXSIZE = 150

        self.weights = []
        CUDA=True
        if(CUDA):
            #weight initialization
            max_, min_ = 0.1, 0
            self.depWeights = {}
            self.depBias = {}
            for x in self.deps: 
                self.depWeights[x] = (torch.rand((self.emb_dim, self.d_model)) * np.sqrt(2/7)).cuda()
                self.depBias[x] = torch.ones((1,self.emb_dim)).cuda()
            self.weights = [(torch.rand((self.emb_dim, self.d_model)) * np.sqrt(2/7)).cuda() for x in range(MAXSIZE)]
            self.bias = [torch.ones((1,self.emb_dim)).cuda() for x in range(MAXSIZE)]
        else:
            #weight initialization
            max_, min_ = 0.1, 0
            self.depWeights = {}
            self.depBias = {}
            for x in self.deps: 
                self.depWeights[x] =  torch.rand((self.emb_dim, self.d_model)) * np.sqrt(2/self.emb_dim)
                self.depBias[x] =  torch.ones((1,self.emb_dim)) 
            self.weights = [torch.rand((self.emb_dim, self.d_model)) * np.sqrt(2/self.emb_dim) for x in range(MAXSIZE)]
            self.bias = [torch.ones((1,self.emb_dim)) for x in range(MAXSIZE)]

    def forward(self, input, dependencies, CUDA):
        """
        input: (embeddings, List of attached dependencies)
        
        embedding size: self.embSize x <words in sentence>
        """
        #print(input.size(),(len(dependencies),self.emb_dim))
        assert input.size() == (len(dependencies),self.emb_dim)

        tensors = []
        for i,x in enumerate(dependencies):
            token = input[i].unsqueeze(0)
            assert token.size() == (1,1024)
            #print(token.size(),self.weights[i].size(),self.bias[i].size())
            accum = token @ self.weights[i] + self.bias[i]
            assert accum.size() == (1,1024)
            for dep in x:
                accum += token @ self.depWeights[dep] + self.depBias[dep]
                #print(accum.size())
                assert accum.size() == (1,1024)
            tensors.append(accum)
        rel = torch.nn.ReLU()
        out = rel(torch.stack(tensors).squeeze(1))
        #print(out.size(),input.size())
        assert out.size() == input.size()
        return out
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        self.load_state_dict(torch.load(filepath))


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for tokenList in testData:

                input = tokenList

                output_task, output_pos = self(input, CUDA)
                pred = predict_class(output_task, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)

        return all_predictions


class Sequence_Tagger_GCNN(nn.Module):
    def __init__(self,word_to_idx,tag_to_ix,pos_to_ix,pretrained_emb,emb_dim,d_model,emb_type,sys_path):
        super().__init__()

        #self.emb=Embedding(emb_dim,len(word_to_idx),pretrained_emb,word_to_idx)
        options_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_options.json")
        weight_file = ""
        
        if emb_type == "original":
            self.emb = "original"
            weight_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        else:
            self.emb = "pubmed"
            weight_file = os.path.join(sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")

        
        self.word_to_idx = word_to_idx
        self.tag_to_ix = tag_to_ix
        self.pos_to_ix = pos_to_ix

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)
        #self.gcnn1 = GCNNLayer(emb_dim,d_model)
        #self.gcnn2 = GCNNLayer(emb_dim,d_model)
        #self.gcnn3 = GCNNLayer(emb_dim,d_model)
        
        #self.gcnn4 = GCNNLayer(emb_dim,d_model)
        self.tagger = LSTMTagger(emb_dim,d_model,len(tag_to_ix),True)
        #self.tagger2 = LSTMTagger(emb_dim//4,d_model//4,len(tag_to_ix),True)
        #self.tagger3 = LSTMTagger(emb_dim//8,d_model//8,len(tag_to_ix),True)
        #self.tagger4 = LSTMTagger(emb_dim,d_model//16,len(tag_to_ix),True)



    def forward(self, input, CUDA):
        tokens, dep = input
        x = self.elmo(Var(batch_to_ids([tokens]),CUDA))
        x = x["elmo_representations"][0]
        #pass1 = self.gcnn1(x.squeeze(),dep,CUDA)
        #pass2 = self.gcnn2(pass1.squeeze(),dep,CUDA)
        #pass3 = self.gcnn3(pass2.squeeze(),dep,CUDA)
        #pass4 = self.gcnn4(pass3.squeeze(),dep,CUDA)

        #print("size x,x.squeeze:",x.size(),x.squeeze().size())
        predictions, lstm_out = self.tagger(x.squeeze())

        return predictions
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        self.load_state_dict(torch.load(filepath))


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for tokenList in testData:

                input = tokenList

                output_task, output_pos = self(input, CUDA)
                pred = predict_class(output_task, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)

        return all_predictions



