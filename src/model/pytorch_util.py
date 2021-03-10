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
from torch.nn import CrossEntropyLoss


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

class RobertaTokenClassifier(nn.Module):
    def __init__(self,model_name,tag_to_ix):
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.num_labels = len(self.tag_to_ix)
        self.roberta = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,add_prefix_space=True)
        self.config = AutoConfig.from_pretrained(model_name)
        print(type(self.tokenizer))
        print(type(self.roberta))
        print(type(self.config))
        
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        

        try:
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        except AttributeError:
            self.dropout = nn.Dropout(0.2)



    def forward(self, input,labels):

        outputs = self.roberta(input['input_ids'].cuda())

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
        return (loss,logits) if loss is not None else output
        # return ((loss,) + output) if loss is not None else output

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        save = {"state_dict": self.state_dict(), "tag_to_ix": self.tag_to_ix}
        torch.save(save,filepath)


    def sm(self,logits,tokens,CUDA):
        inv_dic = get_inv(self.tag_to_ix)
        smx = torch.softmax(logits.squeeze(0),1)
        pred = predict_class(smx, CUDA)

        pred=pred[1:-1]
        pred=id_to_tag(pred,inv_dic)
        pred=contract_annotations(tokens,pred,self.tokenizer)
        pred=merge_annotations(pred)

        return pred



    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        self.eval()
        

        inv_dic = get_inv(self.tag_to_ix)
        
        with torch.no_grad():
            all_predictions = []
            for tokenList in testData:
                

                input= tokenList
                input=self.tokenizer(input,return_tensors="pt",is_split_into_words=True,add_special_tokens=True)
                prediction_logits=self(input,None)


                pred = predict_class(torch.softmax(prediction_logits[0][0],1), CUDA)

                pred=pred[1:-1]
                pred=id_to_tag(pred,inv_dic)

                pred=contract_annotations(tokenList,pred,self.tokenizer)
                pred=merge_annotations(pred)

                all_predictions.append(pred)

        return all_predictions



class BERT_SequenceTagger(nn.Module):
    def __init__(self,model_name,num_labels,tag_to_ix):
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.num_labels=num_labels
        self.bert_model=AutoModel.from_pretrained(model_name)
        config=AutoConfig.from_pretrained(model_name)
        
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        except AttributeError:
            self.dropout = nn.Dropout(0.2)



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
        self.tag_to_ix = temp["tag_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        save = {"state_dict": self.state_dict(), "tag_to_ix": self.tag_to_ix}
        torch.save(save,filepath)


    def sm(self,logits,tokens,CUDA):
        inv_dic = get_inv(self.tag_to_ix)
        smx = torch.softmax(logits.squeeze(0),1)
        pred = predict_class(smx, CUDA)

        pred=pred[1:-1]
        pred=id_to_tag(pred,inv_dic)
        pred=contract_annotations(tokens,pred,self.tokenizer)
        pred=merge_annotations(pred)

        return pred


    def predict(self,testData,CUDA):
        """
        Preditions of the model on testdata
        """
        self.eval()
        

        inv_dic = get_inv(self.tag_to_ix)
        
        with torch.no_grad():
            all_predictions = []
            for tokenList in testData:
                

                input= tokenList
                input=self.tokenizer(input,return_tensors="pt",is_split_into_words=True,add_special_tokens=True)
                prediction_logits=self(input,None)


                pred = predict_class(torch.softmax(prediction_logits[0][0],1), CUDA)

                pred=pred[1:-1]
                pred=id_to_tag(pred,inv_dic)

                pred=contract_annotations(tokenList,pred,self.tokenizer)
                pred=merge_annotations(pred)

                all_predictions.append(pred)

        return all_predictions


class BERT_Matcher(nn.Module):
    def __init__(self,model_name,num_labels,tag_to_ix):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tag_to_ix = tag_to_ix
        self.num_labels = num_labels
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.rel = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        d_model = config.hidden_size
        emb_dim = config.hidden_size

        self.out = nn.Linear(d_model,len(tag_to_ix))

        
    def forward(self,input,labels):

        outputs=self.bert_model(input['input_ids'].cuda())
        sequence_output = outputs[0]
        input['attention_mask']=None

        sequence_output = self.dropout(sequence_output)
        logits = self.out(sequence_output[:,0,:].view(1,-1))


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


        return loss, logits 

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        save = {"state_dict": self.state_dict(), "tag_to_ix": self.tag_to_ix}
        torch.save(save,filepath)

    def predictSys(self,input,CUDA):
        self.eval()
        input = self.tokenizer([input],return_tensors="pt",is_split_into_words=True,add_special_tokens=True,padding=True)
        return self.predict(input,CUDA)



    def predict(self,input,CUDA):
        """
        Predictions of the model on testdata
        """
        self.eval()
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []

            _, logits = self(input,None)
            pred = F.softmax(logits, dim=1)
            _, maxIndices = torch.max(pred,1)

            return inv_dic[maxIndices.cpu().detach().numpy()[0]]













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

        self.rel = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        self.lstm = nn.LSTM(emb_dim, d_model,dropout=0,bidirectional=False)
        self.ff1 = nn.Linear(d_model,d_model)
        self.ff2 = nn.Linear(d_model,d_model)
        self.ff3 = nn.Linear(d_model,int(d_model/2))
        self.out = nn.Linear(int(d_model/2),len(tag_to_ix))


    



    def forward(self, input, CUDA):
        #print(input)
        try:
            x = self.elmo(Var(batch_to_ids([input]),CUDA))
            x = x["elmo_representations"][0]
        except IndexError:
            print(input)
            exit(0)
        
        lstm_out = self.lstm(x)
        if lstm_out[0].shape[1] == 1:
            lstm_finalLayer = lstm_out[0].squeeze()
            lstm_finalLayer = lstm_finalLayer.unsqueeze(0)
        else:
            lstm_finalLayer = lstm_out[0].squeeze()[-1:,]

        #print(lstm_finalLayer.shape)
        lstm_finalLayer = lstm_finalLayer

        ff1 = self.drop(self.rel(self.ff1(lstm_finalLayer)))
        ff2 = self.drop(self.rel(self.ff2(ff1)))
        ff3 = self.drop(self.rel(self.ff3(ff2)))
        linear_out = self.out(ff3)

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

        self.eval()

        with torch.no_grad():
            all_predictions = []
            for tokenList in testData:

                input = tokenList
                if input == []:
                    continue

                _,prediction = self(input, CUDA)
                pred = predict_class(prediction, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)

        return all_predictions

class BERT_Matcher_plus_path(nn.Module):

    def __init__(self,model_name,tag_to_ix,pathModel):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tag_to_ix = tag_to_ix
        self.num_labels = len(tag_to_ix)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        d_model = config.hidden_size
        emb_dim = config.hidden_size

        self.pathModel = pathModel

        self.out = nn.Linear((d_model*2)+self.pathModel.d_model,len(tag_to_ix))

        

    def forward(self,nounBatch,quantBatch,pathList,pathLens,labels,CUDA):
        quantBatch['attention_mask']=None
        nounBatch['attention_mask']=None

        quant_out = self.bert_model(quantBatch['input_ids'].cuda())[0][:,0,:]
        noun_out = self.bert_model(nounBatch['input_ids'].cuda())[0][:,0,:]
        _, path_out = self.pathModel(pathList,pathLens,CUDA)

        
        together = torch.cat((quant_out,path_out,noun_out),1)

        together
        logits = self.out(together)


        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if quantBatch['attention_mask'] is not None:
                active_loss = quantBatch['attention_mask'].view(-1).cuda() == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # output = (logits,) + outputs[2:]
        # return ((loss,) + output) if loss is not None else output
        return loss, logits 

    def loadFromDict(self,temp):
        self.tag_to_ix = temp["tag_to_ix"]
        self.load_state_dict(temp["state_dict"])
        self.pathModel.loadFromDict(temp["pathModel"])

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        self.loadFromDict(torch.load(filepath))

    def save(self,filepath):
        save = {"state_dict": self.state_dict(), "tag_to_ix": self.tag_to_ix, "pathModel":self.pathModel.getSaveDict()}
        torch.save(save,filepath)

    def predictSys(self,nounBatch,quantBatch,pathList,CUDA):
        def padInput(pathList):
            m=0
            for depList in pathList:
                m = max(m, len(depList))

            pathLens = []
            for i,depList in enumerate(pathList):
                pathLens.append(len(depList))
                if len(depList) <  m:
                    pathList[i] = depList + ["pad" for x in range(m-len(depList))]
            
            return pathList,pathLens

        quantBatch = self.tokenizer([quantBatch],return_tensors="pt",is_split_into_words=True,add_special_tokens=True,padding=True)
        nounBatch = self.tokenizer([nounBatch],return_tensors="pt",is_split_into_words=True,add_special_tokens=True,padding=True)
        pathList,pathLens = padInput([pathList])
        temp = self.predict(
            nounBatch=nounBatch,
            quantBatch=quantBatch,
            pathList=pathList,
            pathLens=pathLens,
            CUDA=CUDA)

        return temp


    def predict(self,nounBatch,quantBatch,pathList,pathLens,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():

            _, logits = self(nounBatch,quantBatch,pathList,pathLens,None,CUDA)
            #print(logits)
            pred = F.log_softmax(logits, dim=1)

            _, maxIndices = torch.max(pred,1)
            #print(maxIndices)

            all_predictions = [inv_dic[y] for y in maxIndices.cpu().detach().numpy()]

        return all_predictions



class PathRepresentation(nn.Module):


    def __init__(self,dep_to_ix,tag_to_ix,emb_dim=50,d_model=50):
        """
        tag_to_ix: dictionary mapping all possible output classes to numbers
        dep_to_ix: dictionary mapping all possible dependencies to numbers
        d_model: dimension of the hidden layer of the LSTM and, therefore, its output size
        emb_dim: dimension of the dependency embeddings you wish to create 
        """
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.dep_to_ix = dep_to_ix
        self.d_model = d_model

        self.rel = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        self.depEmbedding = nn.Embedding(len(dep_to_ix), emb_dim)
        self.lstm = nn.LSTM(emb_dim,d_model,dropout=0,bidirectional=False)
        self.ff1 = nn.Linear(d_model,d_model)
        self.ff2 = nn.Linear(d_model,d_model)
        self.ff3 = nn.Linear(d_model,d_model)
        self.out = nn.Linear(d_model,len(tag_to_ix))


    def forward(self, input, pathLens, CUDA):
        """
        input: a path of dependecies you wish to represent
        """
        def getLstmLast(ip,pathLens):
            lstm_out = self.lstm(ip)[0]
            accum = []
            for x in range(lstm_out.shape[0]):
                accum.append(lstm_out[x,pathLens[x]-1,:].view(1,-1))
            return torch.cat(accum,0)

        tensor = None
        for x in input:
            if tensor == None:
                tensor = convert_2_tensor(x,self.dep_to_ix,torch.long,CUDA).unsqueeze(0)
            else:
                temp = convert_2_tensor(x,self.dep_to_ix,torch.long,CUDA).unsqueeze(0)
                tensor = torch.cat((tensor,temp))


        pathEmb = self.depEmbedding(tensor)
        lstm_out = self.drop(getLstmLast(pathEmb,pathLens))
        ff1 = self.drop(self.rel(self.ff1(lstm_out)))
        ff2 = self.drop(self.rel(self.ff2(ff1)))
        ff3 = self.drop(self.rel(self.ff3(ff2)))
        # lstm_out = getLstmLast(pathEmb,pathLens)
        # ff1 = self.ff1(lstm_out)
        # ff2 = self.ff2(ff1)
        # ff3 = self.ff3(ff2)
        output = self.out(ff3)
        return  output, ff3

    def loadFromDict(self,temp):
        """
        Load a cached model form a pre-loaded dictionary
        """
        self.tag_to_ix = temp["tag_to_ix"]
        self.dep_to_ix = temp["dep_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def getSaveDict(self):
        return {
                "dep_to_ix":self.dep_to_ix,
                "tag_to_ix":self.tag_to_ix,
                "state_dict": self.state_dict()
            }
        

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        self.loadFromDict(torch.load(filepath))


    def save(self,filepath):
        """
        save model to a binary file
        """
        torch.save(self.getSaveDict(),filepath)


    def predict(self,input,pathLens,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            
            out,_ = self(input,pathLens,CUDA)
            prediction = F.softmax(out, dim=1)
            _,maxIndices = torch.max(prediction,1)
            all_predictions = [inv_dic[x] for x in maxIndices.cpu().detach().numpy()]
            
        return all_predictions

    def predictSys(self,input,CUDA):
        return self.predict([input],[len(input)],CUDA)

class Elmo_paths(nn.Module):

    def loadElmo(self):
        options_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_options.json")
        
        if self.emb == "original":
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        else:
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)


    def __init__(self,dep_to_ix,tag_to_ix,emb_dim,d_model,emb_type,sys_path):
        super().__init__()
        self.emb = emb_type
        self.sys_path = sys_path
        self.tag_to_ix = tag_to_ix
        self.dep_to_ix = dep_to_ix
        self.d_model = d_model

        self.loadElmo()

        self.depEmbedding = nn.Embedding(len(dep_to_ix), 50)
        self.lstm = nn.LSTM(50,50,dropout=0,bidirectional=False)

        self.feedforward = nn.Linear(d_model + 50,int(d_model/2))
        self.out = nn.Linear(int(d_model/2),len(tag_to_ix))


    



    def forward(self, input, paths, CUDA):
        """
        input: a tokenized sentence ["A","yellow",...,"."]

        paths: a dict of lists of lists of paths, one entry for each cd in a sentence 
        containings the corresponding paths from that cd to each word in that sentence
        """
        def getLstmLast(ip):
            lstm_out = self.lstm(ip)
            if lstm_out[0].shape[1] == 1:
                lstm_finalLayer = lstm_out[0].squeeze()
                lstm_finalLayer = lstm_finalLayer.unsqueeze(0)
            else:
                lstm_finalLayer = lstm_out[0].squeeze()[-1:,]

            return lstm_finalLayer
        #print(input)
        x = self.elmo(Var(batch_to_ids([input]),CUDA))
        x = x["elmo_representations"][0]
        x = x.squeeze(0)

        temp = []
        for depPathList in paths:
            sentRep = None
            for i,depPath in enumerate(depPathList):
                pathRep = self.depEmbedding(convert_2_tensor(depPath,self.dep_to_ix,torch.long,CUDA)).unsqueeze(0) 
                
                # for dep in depPath:
                #     if pathRep == None:
                #         pathRep = self.depEmbedding(self.dep_to_ix[dep])
                #     else:
                #         pathRep = torch.cat((pathRep,self.depEmbedding(self.dep_to_ix[dep])))
                if sentRep == None:
                    sentRep = getLstmLast(pathRep)
                else:
                    tempOut = getLstmLast(pathRep)
                    #print("forward",sentRep.shape,tempOut.shape, pathRep.shape,depPathList[i])
                    sentRep = torch.cat((sentRep,tempOut))
            temp.append(sentRep)

        #print("sentRep",temp[0].shape)
        #print("x",x.shape)
        #is this maintaining grad? 
        temp = [torch.cat((v,x.clone().detach().requires_grad_(True)),1) for v in temp]

        accumPred = []
        accumOut = []
        for sentenceRepresentation in temp:
            linear_out = self.feedforward(sentenceRepresentation)
            linear_out = self.out(linear_out)
            pred = F.softmax(linear_out, dim=1)
            accumOut.append(linear_out)
            accumPred.append(pred)

        return torch.stack(accumPred,0),torch.stack(accumOut,0)
    

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.dep_to_ix = temp["dep_to_ix"]
        self.emb = temp["emb"]
        self.sys_path = temp["sys_path"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        """
        save model to a binary file
        """
        temp = {
            "dep_to_ix":self.dep_to_ix,
            "tag_to_ix":self.tag_to_ix,
            "emb":self.emb,
            "sys_path":self.sys_path,
            "state_dict": self.state_dict()
        }
        torch.save(temp,filepath)


    def predict(self,input,paths,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []

            _,prediction = self(input,paths,CUDA)
            # print(prediction.shape)
            _,maxIndices = torch.max(prediction,2)
            # print(prediction)
            # print(maxIndices)
            for x in [z.cpu().detach().numpy() for z in maxIndices]:
                all_predictions.append([inv_dic[y] for y in x])

            
        return all_predictions





class WordPath(nn.Module):
    

    def loadElmo(self):
        options_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_options.json")
        
        if self.emb == "original":
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
        else:
            weight_file = os.path.join(self.sys_path,"elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5")

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)


    def __init__(self,dep_to_ix,tag_to_ix,emb,sys_path,emb_dim=50,d_model=50):
        """
        Description: this module attemps to classify paths from 
        quantities to other annotations from the same datapoint. 
        However, here we append elmo word embeddings to each dependency 
        embedding 
        annotation this is different because  


        tag_to_ix: dictionary mapping all possible output classes to numbers
        dep_to_ix: dictionary mapping all possible dependencies to numbers
        d_model: dimension of the hidden layer of the LSTM and, therefore, its output size
        emb_dim: dimension of the dependency embeddings you wish to create 
        """
        super().__init__()
        self.tag_to_ix = tag_to_ix
        self.dep_to_ix = dep_to_ix
        self.d_model = emb_dim + 1024
        self.sys_path = sys_path
        self.emb = emb

        self.loadElmo()

        self.depEmbedding = nn.Embedding(len(dep_to_ix), emb_dim)
        self.lstm = nn.LSTM(self.d_model,self.d_model,dropout=0,bidirectional=False)
        self.ff1 = nn.Linear(self.d_model,self.d_model)
        self.ff2 = nn.Linear(self.d_model,self.d_model)
        self.ff3 = nn.Linear(self.d_model,self.d_model)
        self.out = nn.Linear(self.d_model,len(tag_to_ix))





    def forward(self, inputPath, inputWords, pathLens, CUDA):
        """
        input: a path of dependecies you wish to represent
        """

        

        def getLstmLast(ip,pathLens):
            lstm_out = self.lstm(ip)[0]
            accum = []
            for x in range(lstm_out.shape[0]):
                accum.append(lstm_out[x,pathLens[x]-1,:].view(1,-1))
            return torch.cat(accum,0)

        tensor = None
        for x in inputPath:
            if tensor == None:
                tensor = convert_2_tensor(x,self.dep_to_ix,torch.long,CUDA).unsqueeze(0)
            else:
                tensor = torch.cat((tensor,convert_2_tensor(x,self.dep_to_ix,torch.long,CUDA).unsqueeze(0)))

        
        pathEmb = self.depEmbedding(tensor)
        # print(pathEmb.shape)
        elmo_out = self.elmo(Var(batch_to_ids(inputWords),CUDA))["elmo_representations"][0]
        cat = torch.cat((pathEmb,elmo_out),2)


        lstm_out = getLstmLast(cat,pathLens)
        ff1 = self.ff1(lstm_out)
        ff2 = self.ff2(ff1)
        ff3 = self.ff3(ff2)
        output = self.out(ff3)
        return  output, ff3
        

    def load(self,filepath):
        """
        Load a cached model form a file
        """
        temp = torch.load(filepath)
        self.tag_to_ix = temp["tag_to_ix"]
        self.dep_to_ix = temp["dep_to_ix"]
        self.load_state_dict(temp["state_dict"])

    def save(self,filepath):
        """
        save model to a binary file
        """
        temp = {
            "dep_to_ix":self.dep_to_ix,
            "tag_to_ix":self.tag_to_ix,
            "state_dict": self.state_dict()
        }
        torch.save(temp,filepath)


    def predict(self,inputPath,inputWords,pathLens,CUDA):
        """
        Preditions of the model on testdata
        """
        inv_dic = get_inv(self.tag_to_ix)

        with torch.no_grad():
            all_predictions = []
            
            out,_ = self(inputPath,inputWords,pathLens,CUDA)
            prediction = F.softmax(out, dim=1)
            _,maxIndices = torch.max(prediction,1)
            all_predictions = [inv_dic[x] for x in maxIndices.cpu().detach().numpy()]
            
        return all_predictions


#from utils import *
class BERT_SequenceTagger2(nn.Module):
    
    def __init__(self, model_name, tag_to_ix, modif_to_ix):
        super().__init__()
        self.num_labels = len(tag_to_ix)
        self.bert_model = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, len(tag_to_ix))
        self.modifier_classifier = nn.Linear(config.hidden_size, len(modif_to_ix))
        self.w_att = nn.Linear(config.hidden_size, 1)
        self.num_modif = len(modif_to_ix)
        self.inv_dic=get_inv(tag_to_ix)
        self.modif_inv_dic=get_inv(modif_to_ix)


    def predict(self, input, CUDA):
        # Inout=lists of tokens
        # Output= predicted sapn tags, predicted modifiers for each predicted QA
        orig_tokens = input
        with torch.no_grad():
            input = self.tokenizer(input, return_tensors="pt", is_split_into_words=True, add_special_tokens=True)
            _, _, span_logits, modif_pred = self(input, None, None, None,CUDA)
            pred = predict_class(torch.softmax(span_logits[0], 1), CUDA)
            pred = pred[1:-1]
            pred = id_to_tag(pred, self.inv_dic)
            pred = contract_annotations(orig_tokens, pred, self.tokenizer)
            span_pred = merge_annotations(pred)
    
            new_modif = None
            if modif_pred is not None:
                modif_pred = predict_multi_class(modif_pred, CUDA)
                modif_pred = id_to_tag_multi_class(modif_pred, self.modif_inv_dic)
    
                new_modif = list()
                for prediction in modif_pred:
                    if len(prediction) == 0:
                        new_modif.append(['Nomod'])
                    elif 'Nomod' in prediction:
                        new_modif.append(['Nomod'])
                    else:
                        new_modif.append(prediction)

        return span_pred, new_modif

    def forward(self, input, labels, modif, modif_spans,CUDA):

        outputs = self.bert_model(input['input_ids'].cuda())
        sequence_output = outputs[0]

        # spans
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

        # modifier
        modif_pred = None
        if modif is not None:
            loss_fct2 = nn.BCELoss()
            tensor_list = list()
            for span in modif_spans:
                tens_for_span = sequence_output.squeeze()[span[0] + 1:span[1] + 1]
                at_score = self.w_att(tens_for_span)
                at_score = torch.softmax(at_score, 0)
                at_score = torch.transpose(at_score, 1, 0)
                vecotrized = torch.matmul(at_score, tens_for_span)
                tensor_list.append(vecotrized)
            modif_loss = None
            if len(tensor_list) > 0:
                tens_for_modif = torch.stack(tensor_list).cuda()
                modif_pred = self.modifier_classifier(tens_for_modif.squeeze(1))
                modif_pred = torch.sigmoid(modif_pred)
                modif_label = list()
                for modifier in modif:
                    cur_labels = [0] * self.num_modif
                    labs = modifier[1]
                    for lab in labs:
                        cur_labels[modif_to_ix[lab]] = 1
                    modif_label.append(cur_labels)

                modif_label = torch.tensor(modif_label, dtype=torch.float).cuda()
                modif_loss = loss_fct2(modif_pred, modif_label)
        else:
            modif_loss = None
            span_preds = predict_class(torch.softmax(logits[0], 1), CUDA)
            span_preds = id_to_tag(span_preds, self.inv_dic)
            qa_spans = get_qa_spans(span_preds)
            tensor_list = list()
            for qa_span in qa_spans:
                tens_for_span = sequence_output.squeeze()[qa_span[0]:qa_span[1]]
                at_score = self.w_att(tens_for_span)
                at_score = torch.softmax(at_score, 0)
                at_score = torch.transpose(at_score, 1, 0)
                vecotrized = torch.matmul(at_score, tens_for_span)
                tensor_list.append(vecotrized)
            tens_for_modif = list()
            if len(tensor_list) > 0:
                tens_for_modif = torch.stack(tensor_list).cuda()
                modif_pred = self.modifier_classifier(tens_for_modif.squeeze(1))
                modif_pred = torch.sigmoid(modif_pred)

        return (loss, modif_loss, logits, modif_pred)
