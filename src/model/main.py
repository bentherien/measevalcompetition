import argparse
import json
import latex
import os 
import pickle
import pprint
import numpy as np

from helpers import *
from load_util import load_sample,load_sample_dep
from tqdm import tqdm
from math import floor
from sklearn_crfsuite.metrics import flat_classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_util import Embedding, LSTMTagger, Sequence_Tagger, Sequence_Tagger_GCNN
from torch import optim
CUDA = True

def run_train(args):
    global CUDA

    print(args)

    test = load_sample_dep(os.path.join(args.sys_path,"data","test_"+args.data+".tsv"))
    train = load_sample_dep(os.path.join(args.sys_path,"data","train_"+args.data+".tsv"))
    all_data = test + train

    word_to_ix = {}
    tag_to_ix = {}
    pos_to_ix={}

    for sample in all_data:
        for token in sample.tokens:
            if token not in word_to_ix:
                word_to_ix[token] = len(word_to_ix)

        for tag in sample.target:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

        for pos in sample.pos:
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)

    pretrained_emb=None

    if CUDA:
        model = Sequence_Tagger_GCNN(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,args.embeddings,args.sys_path).cuda()
    else:
        model = Sequence_Tagger_GCNN(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,args.embeddings,args.sys_path)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    accum = []

    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal tag_to_ix
        nonlocal model 
        nonlocal accum
        nonlocal args

        inv_dic = get_inv(tag_to_ix)

        with torch.no_grad():
            all_predictions = list()
            all_gold = list()
            for sample in test:

                input = (sample.tokens,sample.deps)

                output_task = model(input, CUDA)
                pred = predict_class(output_task, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)
                all_gold.append(sample.target)
            

            temp = latex.getSummary(
                epoch=epoch,
                fold=args.fold,
                embeddingType=model.emb,
                learningRate=args.learning_rate,
                report = flat_classification_report(all_gold,all_predictions),
                title = args.run_title)


            if not silent:
                pp = pprint.PrettyPrinter(indent = 4)
                pp.pprint(temp)

            accum += temp



    for epoch in range(args.epoch_num):


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for i in tqdm(range(len(train))):
            

            model.zero_grad()

            pos = train[i].pos
            pos = convert_2_tensor(pos,pos_to_ix,torch.long,CUDA)

            target = train[i].target
            target = convert_2_tensor(target, tag_to_ix, torch.long, CUDA)

            input = (train[i].tokens,train[i].deps)
            output_task = model(input, CUDA)

            loss = loss_function(output_task, target)
            loss.backward(retain_graph = True)
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        if epoch % 1 != 0:
            optimizer.lr = optimizer.lr/2



        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)

    savePath = os.path.join(args.sys_path,f"{args.model_name}.pt")
    torch.save(model.state_dict(),savePath)

    
    

    if not os.path.isfile("run_logs.json"):
        json.dump([accum],open("run_logs.json","w",encoding="utf-8"),indent=3)
    else:
        try:
            read = json.load(open("run_logs.json","r",encoding="utf-8"))
            read.append(accum)
            json.dump(read,open("run_logs.json","w",encoding="utf-8"),indent=3)
        except json.decoder.JSONDecodeError:
            json.dump(accum,open("run_logs.json","w",encoding="utf-8"),indent=3)

    latex.toLatex(accum)

    exit(0)

def run_test(args):
    global CUDA

    print(args)

    #start
    test = load_sample(os.path.join(args.sys_path,"data","test_"+args.data+".tsv"))
    train = load_sample(os.path.join(args.sys_path,"data","train_"+args.data+".tsv"))
    all_data = test + train


    word_to_ix = {}
    tag_to_ix = {}
    pos_to_ix = {}

    for sample in all_data:
        for token in sample.tokens:
            if token not in word_to_ix:
                word_to_ix[token] = len(word_to_ix)

        for tag in sample.target:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

        for pos in sample.pos:
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)


    pretrained_emb=None

    if CUDA:
        model = Sequence_Tagger(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,args.embeddings,args.sys_path).cuda()
    else:
        model = Sequence_Tagger(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,args.embeddings,args.sys_path)

    model.load(args.saved_model)

    testSize, trainSize = getSizeData(test), getSizeData(train)

    accum = []
    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal tag_to_ix
        nonlocal model 
        nonlocal accum
        nonlocal args

        inv_dic = get_inv(tag_to_ix)

        with torch.no_grad():
            all_predictions = list()
            all_gold = list()
            for sample in test:

                input = (sample.tokens,sample.deps)

                output_task = model(input, CUDA)
                pred = predict_class(output_task, CUDA)

                pred = id_to_tag(pred,inv_dic)

                all_predictions.append(pred)
                all_gold.append(sample.target)
            

            temp = latex.getSummary(
                epoch=epoch,
                fold=args.fold,
                embeddingType=model.emb,
                learningRate=args.learning_rate,
                report = flat_classification_report(all_gold,all_predictions),
                title = args.run_title)


            if not silent:
                pp = pprint.PrettyPrinter(indent = 4)
                pp.pprint(temp)

            accum += temp

    check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)

    exit(0)




def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    subparser.add_argument("--output-path", "-o", default="output.tex")
    subparser.add_argument("--sys-path", "-sp", default="generatedData/")
    subparser.add_argument("--data", "-d", default="allS")
    subparser.add_argument("--model-name", "-n",  default="DefaultModelName")
    subparser.add_argument("--epoch-num", "-e",type=int, default=2)
    subparser.add_argument("--fold", "-f", type=int, default=1)
    subparser.add_argument("--learning-rate", "-l", type=float, default= 0.00042)
    subparser.add_argument("--embeddings", "-emb", type=str, default="original")
    subparser.add_argument("--run-title", "-rt", type=str, default="original")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--output-path", "-o", default="output.tex")
    subparser.add_argument("--sys-path", "-sp", default="generatedData/")
    subparser.add_argument("--model-name", "-n",  default="DefaultModelName")
    subparser.add_argument("--saved-model", "-s",  default="path to saved model")
    subparser.add_argument("--data", "-d", default="allS")
    subparser.add_argument("--epoch-num", "-e",type=int, default=2)
    subparser.add_argument("--fold", "-f", type=int, default=1)
    subparser.add_argument("--learning-rate", "-l", type=float, default= 0.00042)
    subparser.add_argument("--embeddings", "-emb", type=str, default="original")

    

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()






