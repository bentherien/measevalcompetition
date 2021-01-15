# _*_ coding: utf-8 _*_
import os
import numpy as np

from helpers import *
from allennlp.modules.elmo import Elmo,batch_to_ids
from transformers import AutoModel, AutoTokenizer, AutoConfig

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

def get_inv(my_map):
    return {v: k for k, v in my_map.items()}



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
    def __init__(self,emb_dim,d_model,tagSize):
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
                self.depWeights[x] = (torch.rand((self.emb_dim, self.d_model),requires_grad=True) * np.sqrt(2/7)).cuda()
                self.depBias[x] = torch.ones((1,self.emb_dim),requires_grad=True).cuda()
            self.weight = (torch.rand((self.emb_dim, self.d_model),requires_grad=True) * np.sqrt(2/7)).cuda() 
            self.bias = torch.ones((1,self.emb_dim),requires_grad=True).cuda() 

            self.hidden2tag = nn.Linear(d_model , tagSize).cuda()
        else:
            #weight initialization
            max_, min_ = 0.1, 0
            self.depWeights = {}
            self.depBias = {}
            for x in self.deps: 
                self.depWeights[x] =  torch.rand((self.emb_dim, self.d_model),requires_grad=True) * np.sqrt(2/self.emb_dim)
                self.depBias[x] =  torch.ones((1,self.emb_dim),requires_grad=True) 
            self.weight = torch.rand((self.emb_dim, self.d_model),requires_grad=True) * np.sqrt(2/self.emb_dim) 
            self.bias = torch.ones((1,self.emb_dim),requires_grad=True) 
            self.hidden2tag = nn.Linear(d_model , tagSize)

    def forward(self, input, dependencies, CUDA):
        """
        input: (embeddings, List of attached dependencies)
        
        embedding size: self.embSize x <words in sentence>
        """
        #print(input.size(),(len(dependencies),self.emb_dim))
        assert input.size() == (len(dependencies),self.emb_dim)
        selfOut = input @ self.weight + self.bias
        tensors = []
        for i,x in enumerate(dependencies):
            token = input[i].unsqueeze(0)
            assert token.size() == (1,1024)
            #print(token.size(),self.weights[i].size(),self.bias[i].size())
            #accum = token @ self.weights[i] + self.bias[i]
            
            
            accum = selfOut[i].unsqueeze(0)
            assert accum.size() == (1,1024)

            for dep in x:
                accum += token @ self.depWeights[dep] + self.depBias[dep]
                assert accum.size() == (1,1024)

            tensors.append(accum)
        rel = torch.nn.ReLU()
        out = rel(torch.stack(tensors).squeeze(1))
        assert out.size() == input.size()

        #tag_space = self.hidden2tag(lstm_tuple[0].view(len(lstm_tuple[0]), -1))
        tag_space = self.hidden2tag(out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return (tag_scores,out)
    

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
        self.gcnn1 = GCNNLayer(emb_dim,d_model,len(tag_to_ix))
        self.gcnn2 = GCNNLayer(emb_dim,d_model,len(tag_to_ix))
        self.gcnn3 = GCNNLayer(emb_dim,d_model,len(tag_to_ix))
        
        #self.gcnn2 = GCNNLayer(emb_dim,d_model)
        #self.gcnn3 = GCNNLayer(emb_dim,d_model)
        
        #self.gcnn4 = GCNNLayer(emb_dim,d_model)
        #self.tagger = LSTMTagger(emb_dim,d_model,len(tag_to_ix),True)
        #self.tagger2 = LSTMTagger(emb_dim//4,d_model//4,len(tag_to_ix),True)
        #self.tagger3 = LSTMTagger(emb_dim//8,d_model//8,len(tag_to_ix),True)
        #self.tagger4 = LSTMTagger(emb_dim,d_model//16,len(tag_to_ix),True)

        



    def forward(self, input, CUDA):
        tokens, dep = input
        x = self.elmo(Var(batch_to_ids([tokens]),CUDA))
        x = x["elmo_representations"][0]
        predictions,gcnn_out = self.gcnn1(x.squeeze(),dep,CUDA)
        predictions,gcnn_out = self.gcnn2(gcnn_out.squeeze(),dep,CUDA)
        predictions,gcnn_out = self.gcnn3(gcnn_out.squeeze(),dep,CUDA)
        #pass4 = self.gcnn4(pass3.squeeze(),dep,CUDA)

        #print("size x,x.squeeze:",x.size(),x.squeeze().size())
        #predictions, lstm_out = self.tagger(pass1.squeeze())

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

class BERT_SequenceTagger(nn.Module):
    def __init__(self,model_name,num_labels,tag_to_ix):
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.num_labels=num_labels
        self.bert_model=AutoModel.from_pretrained(model_name)
        config=AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)


    def forward(self, input,labels):

        outputs=self.bert_model(input['input_ids'].cuda())

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if input['attention_mask'] is not None:
                active_loss = input['attention_mask'].view(-1).cuda() == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag"]
        self.load_state_dict(temp["state_dict"])


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased')

        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            for tokenList in testData:

                input= tokenList
                input=tokenizer(input,return_tensors="pt",is_split_into_words=True,add_special_tokens=True)
                prediction_logits=self(input,None)


                pred = predict_class(torch.softmax(prediction_logits[0][0],1), CUDA)

                pred=pred[1:-1]
                pred=id_to_tag(pred,inv_dic)


                pred=contract_annotations(tokenList,pred,tokenizer)
                pred=merge_annotations(pred)

                all_predictions.append(pred)

        return all_predictions




class Modifier_Tagger(nn.Module):

    def loadElmo(self):
        options_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_options.json")
        
        if self.emb == "original":
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        else:
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)


    def __init__(self,tag_to_ix,emb_dim,d_model,emb_type,sys_path):
        super().__init__()
        self.emb = emb_type
        self.sys_path = sys_path
        self.tag_to_ix = tag_to_ix
        self.d_model = d_model

        self.loadElmo()

        self.lstm = nn.LSTM(emb_dim, d_model,dropout=0,bidirectional=False)
        # self.linear_1 = nn.Linear(d_model,2*d_model)
        # self.linear_2 = nn.Linear(2*d_model,d_model)
        self.linear_3 = nn.Linear(d_model,int(d_model/2))
        # self.linear_4 = nn.Linear(int(d_model/2),int(d_model/4))
        self.out = nn.Linear(int(d_model/2),len(tag_to_ix))


    



    def forward(self, input, CUDA):
        #print(input)
        x = self.elmo(Var(batch_to_ids([input]),CUDA))
        x = x["elmo_representations"][0]
        
        lstm_out = self.lstm(x)
        if lstm_out[0].shape[1] == 1:
            lstm_finalLayer = lstm_out[0].squeeze()
            lstm_finalLayer = lstm_finalLayer.unsqueeze(0)
            #print("one",lstm_finalLayer.shape)
        else:
            lstm_finalLayer = lstm_out[0].squeeze()[-1:,]
            #print("larger",lstm_finalLayer.shape)

        # linear_out = self.linear_1(lstm_finalLayer)
        # linear_out = self.linear_2(linear_out)
        linear_out = self.linear_3(lstm_finalLayer)
        # linear_out = self.linear_4(linear_out)
        linear_out = self.out(linear_out)

        #print(linear_out.shape)
        predictions = F.softmax(linear_out, dim=1)

        return linear_out, predictions
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.emb = temp["emb"]
        self.sys_path = temp["sys_path"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        """
        save model to a binary file
        """
        temp = {
            "tag_to_ix":self.tag_to_ix,
            "emb":self.emb,
            "sys_path":self.sys_path,
            "state_dict": self.state_dict()
        }
        torch.save(temp,filepath)


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            for tokenList in testData:

                input = tokenList

                _,prediction = self(input, CUDA)
                pred = predict_class(prediction, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)

        return all_predictions

