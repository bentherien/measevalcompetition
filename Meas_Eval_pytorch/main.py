import pickle
import argparse
import json
import latex
import math 
import common
import os 
import numpy as np
import random

from load_util import load_sample
from tqdm import tqdm
from math import floor
from sklearn_crfsuite.metrics import flat_classification_report

import torch
import torch.nn as nn
from pytorch_util import Embedding, LSTMTagger
from torch import optim
from allennlp.modules.elmo import Elmo,batch_to_ids
import torch.nn.functional as F
from torch.autograd import Variable

CUDA = True

def Var(v):
    """
    Checks if program is set to run as cuda 
    """
    if CUDA:
        return Variable(v.cuda())
    else:
        return Variable(v)

def id_to_tag(seq,to_tag):
    """
    res = []

    for ele in seq:
        res.append(to_tag[ele])

    return res
    """
    return [to_tag[e] for e in seq]

def get_inv(my_map):
    """
    Inverts the keys and values in a given dicitonary
    """
    return {v: k for k, v in my_map.items()}

def get_set(all,perm):
    """

    """
    new_set = list()
    for i in range(len(all)):
        if i in perm:
            new_set.append(all[i])
    return new_set

def convert_2_tensor(seq, to_ix, dt):
    """
    """
    if to_ix == None:
        return Var(torch.tensor(seq, dtype=dt))
    else:
        idxs = list()
        for w in seq:
            if w in to_ix:
                idxs.append(to_ix[w])
        return Var(torch.tensor(idxs, dtype=dt))

def predict_class(class_score, CUDA):
    if CUDA:
        class_score = class_score.cpu().detach().numpy()
    else:
        class_score = class_score.detach().numpy()
    classes = list()
    for seq in class_score:
        classes.append(np.argmax(seq))
    return classes

def test(fileObj,test,testSize,trainSize,epoch,avgLoss,fold,silent=True):
    inv_dic=get_inv(tag_to_ix)

    with torch.no_grad():
        all_predictions = list()
        all_gold = list()
        for sample in test:

            input = sample.tokens
            #input = convert_2_tensor(input, word_to_ix, torch.long)
            input = [input]

            output = model(input)
            pred = predict_class(output, CUDA)

            pred=id_to_tag(pred,inv_dic)

            all_predictions.append(pred)
            all_gold.append(sample.target)
        
        if not silent:
            print(flat_classification_report(all_gold,all_predictions))
            print("Dev set stats:",file= fileObj)
            
        sizesTest = "testCount("
        for x in testSize:
            sizesTest += str(x) + ":" + str(testSize[x]) + ","
        sizesTest = sizesTest[:-1] + ")"

        sizesTrain = "trainCount("
        for x in trainSize:
            sizesTrain += str(x) + ":" + str(trainSize[x]) + ","
        sizesTrain = sizesTrain[:-1] + ")"

        latex.writeCReport(
            report = flat_classification_report(all_gold,all_predictions), 
            fileObj = fileObj, 
            label="e-{}data-{}".format(epoch,args.data_path.split("/")[-1]), 
            caption="e-{},dataset-{},avgloss-{},fold-{},model-Elmo:BiLSTM,{},{}".format(epoch,args.model_name,avgLoss,fold,sizesTest,sizesTrain), 
            arrangement="|X|X|X|X|X|", 
            size="300pt",
            epoch=epoch,
            fold=args.fold
        )
        fileObj.flush()
        fileObj.write("\n")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("run")
    subparser.add_argument("--output-path", "-o", default="data/22.auto.clean")
    subparser.add_argument("--data-path", "-d", default="samples_test.tsv")
    subparser.add_argument("--model-name", "-n", required=True)
    subparser.add_argument("--epoch-num", "-e", required=True)
    subparser.add_argument("--fold", "-f", type=int,required=True)

    args = parser.parse_args()
    return args

class Sequence_Tagger(nn.Module):
    def __init__(self, word_to_idx,tag_to_ix,pretrained_emb, emb_dim,d_model):
        super().__init__()

        #self.emb=Embedding(emb_dim,len(word_to_idx),pretrained_emb,word_to_idx)
        options_file = "elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(options_file,weight_file,1,dropout=0)
        self.tagger = LSTMTagger(emb_dim,d_model,len(tag_to_ix),True)

    def forward(self, input):

        #x=self.emb(input)
        va = Var(batch_to_ids(input))
        x = self.elmo(va)
        x = x["elmo_representations"][0]
        x = x.squeeze()

        predictions, lstm_out   =   self.tagger(x)

        return predictions


args = main()

tempFile = os.path.join("output",args.model_name)

if os.path.isfile(tempFile):
    common.f = open(tempFile, "a", encoding="utf-8")
else:
    common.f = open(tempFile, "a", encoding="utf-8")
    latex.tableStart(label="SummaryTable-{}".format(args.model_name), caption="SummaryTable-{}".format(args.model_name), arrangement="|X|X|X|X|X|X|X|", size="325pt")

all_data = load_sample(args.data_path)

word_to_ix = {}
tag_to_ix = {}

for sample in all_data:
    for token in sample.tokens:
        if token not in word_to_ix:
            word_to_ix[token] = len(word_to_ix)

    for tag in sample.target:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

#pretrained_emb = pickle.load(open('glove_wiki.p', 'rb'))

pretrained_emb=None


if CUDA:
    model = Sequence_Tagger( word_to_ix, tag_to_ix ,pretrained_emb, 1024,1024).cuda()
else:
    model = Sequence_Tagger( word_to_ix, tag_to_ix ,  pretrained_emb, 1024,1024)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

statFile = open(args.output_path, "a", encoding="utf-8")
num_epochs = int(args.epoch_num)

def getFolds(fold, data, div = 8):
    if fold < 1  or fold > div:
        print("Incorrect div to fold number encountered")
        return None, None
    testSize = math.floor(len(data)/div)
    beforeTest = data[:(fold-1)*testSize]
    test = data[(fold-1)*testSize:fold*testSize]
    afterTest = data[fold*testSize:]
    return test, beforeTest + afterTest

    


def getSizeData(data):
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
    del freq["o"]
    return freq
print("&".join(["fold","type","QA","ME","MP","QL"])+"\\\\\n\\hline")
for fold in range(1,9):
    testData, train = getFolds(fold, all_data, 8)
    testSize, trainSize = getSizeData(testData), getSizeData(train)
    
    ltemp = [str(fold), "test"]
    for k in ["ME","MP","QL","QA"]:
        ltemp.append(str(testSize[k]))
    print("&".join(ltemp)+"\\\\\n\\hline")
    ltemp = [str(fold), "train"]
    for k in ["ME","MP","QL","QA"]:
        ltemp.append(str(trainSize[k]))
    print("&".join(ltemp)+"\\\\\n\\hline")
exit(0)
#testData, train = getFolds(args.fold, all_data, 8)
#testSize, trainSize = getSizeData(testData), getSizeData(train)

for epoch in range(num_epochs):


    all_loss = list()

    print('Epoch: ' + str(epoch + 1))

    for i in tqdm(range(len(train))):
        

        model.zero_grad()
        input = train[i].tokens

        input = [input]
        #print(input)

        target = train[i].target

        #input = convert_2_tensor(input, word_to_ix, torch.long)

        target = convert_2_tensor(target, tag_to_ix, torch.long)

        output = model(input)

        loss = loss_function(output, target)
        
        loss.backward()
        #Q why are we not using optimizer.zero_grad() at this point, wont we accumulate gradients?
        optimizer.step()

        all_loss.append(loss.cpu().detach().numpy())

    if epoch == 10:
        optimizer.lr = 0.005/2

    print('\n')
    print('Average loss: ' + str(np.mean(all_loss)))
    #print('Average loss: ' + str(np.mean(all_loss)), file=statFile)


    test(statFile, testData, testSize, trainSize, epoch=(epoch+1), avgLoss=np.mean(all_loss), fold=args.fold, silent=True)


common.f.close()
statFile.close()


