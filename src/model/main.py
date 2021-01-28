import argparse
import json
import latex
import os 
import pickle
import pprint
import numpy as np

from helpers import *
from latex import *
from load_util import load_samples
from data import PathData
from tqdm import tqdm
from math import floor
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_util import *
from torch import optim
CUDA = True


def run_train(args):
    if args.type == "modifier":
        run_train_modifier(args)
    elif args.type == "bert_token":
        run_train_bert_token(args)
    elif args.type == "bert":
        train_bert(args)
    elif args.type == "paths":
        run_train_paths(args)
    elif args.type == "paths2":
        run_train_paths2(args)
    elif args.type == "paths-word":
        run_train_paths_word(args)
    else:
        run_train_token(args)

def train_bert(args):
    global CUDA

    test = json.load(open(os.path.join(args.sys_path,"data-json","test-{}.json".format(args.data)),"r",encoding="utf-8"))

    train = json.load(open(os.path.join(args.sys_path,"data-json","train-{}.json".format(args.data)),"r",encoding="utf-8"))

    tag_to_ix = {}

    for sample in train+test:
        tag = sample[2]
        try:
            tag_to_ix[tag]
        except KeyError:
            tag_to_ix[tag] = len(tag_to_ix)

    model_name = 'allenai/scibert_scivocab_cased'

    if CUDA:
        model = BERT_Matcher( model_name, len(tag_to_ix),tag_to_ix).cuda()
    else:
        model = BERT_Matcher( model_name, len(tag_to_ix),tag_to_ix)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal model 
        nonlocal args
        nonlocal tokenizer
        #model.eval()

        with torch.no_grad():
            all_predictions = []
            all_gold = []
            inv_dict = get_inv(model.tag_to_ix)
            maxIndices = None
            for span1, span2, target in tqdm(test):
                input = tokenizer([span1,span2],return_tensors="pt",is_split_into_words=True,add_special_tokens=True,padding=True)
                all_gold.append(target)
                
                _, logits = model(input,None)
                pred = F.log_softmax(logits, dim=1)
                

                if maxIndices == None:
                    _, maxIndices = torch.max(pred,1)
                else:
                    _, tempMaxIndices = torch.max(pred,1)
                    maxIndices = torch.cat((maxIndices,tempMaxIndices),0)
            all_predictions = [inv_dict[x] for x in maxIndices.cpu().detach().numpy()]
            
        return classification_report(all_gold,all_predictions)

    maxF1 = 0

    for epoch in range(args.epoch_num):
        model.train()

        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for span1, span2, target in tqdm(train):
            model.zero_grad()
            
            target=convert_2_tensor([target],tag_to_ix,torch.long,CUDA)

            input = tokenizer([span1,span2],return_tensors="pt",is_split_into_words=True,add_special_tokens=True,padding=True)
            loss,_ = model(input,target)

            loss.backward()
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        report = check_test(test,epoch,np.mean(all_loss),1)
        print(report)
        f1 = getLabel(model.tag_to_ix,report,label="macro")

        if f1 > maxF1:
            print("New Max F1: {}".format(f1))
            savePath = os.path.join(args.sys_path,f"{args.model_name}.pt")
            model.save(savePath)
            maxF1 = f1

    exit(0)

def run_train_bert_token(args):
    global CUDA

    def check_test():
        nonlocal tag_to_ix
        nonlocal model
        nonlocal test


        inv_dic=get_inv(tag_to_ix)
        with torch.no_grad():
            all_predictions = list()
            all_gold = list()
            for sample in test:

                input= sample.tokens
                target=sample.targets
                input=tokenizer(input,return_tensors="pt",is_split_into_words=True,add_special_tokens=True)
                prediction_logits=model(input,None)


                pred = predict_class(torch.softmax(prediction_logits[0][0],1), CUDA)

                pred=pred[1:-1]
                pred=id_to_tag(pred,inv_dic)
                #print(pred)


                pred=contract_annotations(sample.tokens,pred,tokenizer)
                try:
                    pred=merge_annotations(pred)
                except ValueError:
                    print(pred)

                assert(len(pred) == len(target))
                all_predictions.append(pred)
                all_gold.append(target)

            # for i,x in enumerate(all_predictions):
            #     assert(len(all_predictions[i])==len(all_gold[i]))
            #     print("pred:",all_predictions[i])
            #     print("gold:",all_gold[i])


            return flat_classification_report(all_gold,all_predictions)

    model_name='allenai/scibert_scivocab_cased'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train = load_samples(os.path.join(args.sys_path,"data-fold{}".format(args.fold),"train_allS.tsv"))
    test = load_samples(os.path.join(args.sys_path,"data-fold{}".format(args.fold),"test_allS.tsv"))


    word_to_ix = {}
    tag_to_ix = {}
    pos_to_ix={}


    for sample in train+test:
        for token in sample.tokens:
            if token not in word_to_ix:
                word_to_ix[token] = len(word_to_ix)

        for tag in sample.targets:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

        for pos in sample.pos:
            if pos not in pos_to_ix:
                pos_to_ix[pos] = len(pos_to_ix)


    if CUDA:
        model = BERT_SequenceTagger( model_name, len(tag_to_ix),tag_to_ix).cuda()
    else:
        model = BERT_SequenceTagger( model_name, len(tag_to_ix),tag_to_ix)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    maxf1 = 0

    for epoch in range(args.epoch_num):


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for i in tqdm(range(len(train))):

            sample=train[i]
            model.zero_grad()
            input = train[i].tokens
            target = train[i].targets
            target = expand_annotations(input,target,tokenizer)
            target.insert(0,'o')
            target.append('o')
            target=convert_2_tensor(target,tag_to_ix,torch.long,CUDA)

            input=tokenizer(input,return_tensors="pt",is_split_into_words=True,add_special_tokens=True)
            loss,_=model(input,target)

            loss.backward()
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        report = check_test()
        print(report)
        f1 = getLabel(model.tag_to_ix,report,label="macro")

        if f1 > maxf1:
            print("Saving Model for new New Max macro F1: {}".format(f1))
            savePath = os.path.join(args.sys_path,f"{args.model_name}fold{args.fold}.pt")
            model.save(savePath)
            maxf1 = f1


        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))



def run_train_paths_word(args):
    global CUDA

    test = json.load(open(os.path.join(args.sys_path,"path-data-2","test-path-data-NOUN.json"),"r",encoding="utf-8"))
    test = PathData(test)

    train = json.load(open(os.path.join(args.sys_path,"path-data-2","train-path-data-NOUN.json"),"r",encoding="utf-8"))
    train = PathData(train)

    all_tags = test.getTags() + train.getTags()

    tag_to_ix = {}
    dep_to_ix = {}

    deps = ['pad','subtok','self','ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', "parataxis", "pcomp", "pobj", "poss", "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp"]
    rdeps = ["r"+x for x in deps]

    all_deps = deps + rdeps + test.getDeps2() + train.getDeps2()

    for dep in all_deps:
        try:
            dep_to_ix[dep]
        except KeyError:
            dep_to_ix[dep] = len(dep_to_ix)

    for tag in all_tags:
        try:
            tag_to_ix[tag]
        except KeyError:
            tag_to_ix[tag] = len(tag_to_ix)


    pretrained_emb=None

    if CUDA:
        model = WordPath(dep_to_ix,tag_to_ix,args.embeddings,args.sys_path,emb_dim=64,d_model=64).cuda()
    else:
        model = WordPath(dep_to_ix,tag_to_ix,args.embeddings,args.sys_path,emb_dim=64,d_model=64)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    accum = []

    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal model 
        nonlocal accum
        nonlocal args


        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for sent in test.sents:
                if sent.targets == []:
                    continue

                def pad(l,pad="pad"):
                    m=0
                    for pathListList in l:
                        for pathList in pathListList:
                            m = max(m, len(pathList))

                    pathLens = []
                    for i,pathListList in enumerate(l):
                        for ii,pathList in enumerate(pathListList):
                            pathLens.append(len(pathList))
                            if len(pathList) <  m:
                                l[i][ii] = pathList + [pad for x in range(m-len(pathList))]
                    return l,pathLens

                
                inputWords = [[[z[1] for z in y] for y in x] for x in sent.samples]
                inputPaths = [[[z[0] for z in y] for y in x] for x in sent.samples]

                inputWords,pathLens = pad(inputWords,"<UNK>")
                inputPaths,pathLens = pad(inputPaths,"pad")
                

                accum = []
                for x in inputPaths:
                    accum += x

                accumWords = []
                for word in inputWords: 
                    accumWords += word

                tg = []
                for target in sent.targets:
                    tg += target

                # print(len(accum),len(tg))
                pred = model.predict(accum,accumWords,pathLens,CUDA)
                assert(len(accum) == len(pred))
                
                all_predictions = all_predictions + pred
                all_gold = all_gold + tg
            # print(all_predictions)
            # print(all_gold)

            # print(len(all_predictions),len(all_gold))

            print(classification_report(all_gold,all_predictions))
            

 
    for epoch in range(args.epoch_num):


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for sent in tqdm(train.sents):
            if sent.targets == []:
                continue

            model.zero_grad()
            tg = []
            for target in sent.targets:
                tg += target

            target = convert_2_tensor(tg, tag_to_ix, torch.long, CUDA)
            # print(target)
            # print(sent.targets)
            #target = torch.stack(target,0)

            def pad(l,pad="pad"):
                    m=0
                    for pathListList in l:
                        for pathList in pathListList:
                            m = max(m, len(pathList))

                    pathLens = []
                    for i,pathListList in enumerate(l):
                        for ii,pathList in enumerate(pathListList):
                            pathLens.append(len(pathList))
                            if len(pathList) <  m:
                                l[i][ii] = pathList + [pad for x in range(m-len(pathList))]
                    return l, pathLens

                
            inputWords = [[[z[1] for z in y] for y in x] for x in sent.samples]
            inputPaths = [[[z[0] for z in y] for y in x] for x in sent.samples]

            inputWords,pathLens = pad(inputWords,"<UNK>")
            inputPaths,pathLens = pad(inputPaths,"pad")

            accumWords = []
            for word in inputWords: 
                accumWords += word

            accum = []
            for x in inputPaths:
                accum += x


            out, _ = model(accum,accumWords,pathLens,CUDA)
            

            loss = loss_function(out, target)
            loss.backward(retain_graph = True)
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        if epoch % 1 != 0:
            optimizer.lr = optimizer.lr/2



        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)

    savePath = os.path.join(args.sys_path,f"{args.model_name}.pt")
    model.save(savePath)

    exit(0)


def run_train_paths2(args):
    global CUDA

    test = json.load(open(os.path.join(args.sys_path,"path-data-2","test-path-data-NOUN-BEST.json"),"r",encoding="utf-8"))
    test = PathData(test)

    train = json.load(open(os.path.join(args.sys_path,"path-data-2","train-path-data-NOUN-BEST.json"),"r",encoding="utf-8"))
    train = PathData(train)

    all_tags = test.getTags() + train.getTags()

    tag_to_ix = {}
    dep_to_ix = {}

    deps = ['pad','subtok','self','ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', "parataxis", "pcomp", "pobj", "poss", "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp"]
    rdeps = ["r"+x for x in deps]

    all_deps = deps + rdeps + test.getDeps2() + train.getDeps2()

    for dep in all_deps:
        try:
            dep_to_ix[dep]
        except KeyError:
            dep_to_ix[dep] = len(dep_to_ix)

    for tag in all_tags:
        try:
            tag_to_ix[tag]
        except KeyError:
            tag_to_ix[tag] = len(tag_to_ix)


    pretrained_emb=None

    if CUDA:
        model = PathRepresentation(dep_to_ix,tag_to_ix,emb_dim=16,d_model=16).cuda()
    else:
        model = PathRepresentation(dep_to_ix,tag_to_ix,emb_dim=16,d_model=16)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    accum = []

    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal model 
        nonlocal accum
        nonlocal args


        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for sent in test.sents:
                if sent.targets == []:
                    continue

                input = [[[z[0] for z in y] for y in x] for x in sent.samples]
                m=0
                for pathListList in input:
                    for pathList in pathListList:
                        m = max(m, len(pathList))

                pathLens = []
                for i,pathListList in enumerate(input):
                    for ii,pathList in enumerate(pathListList):
                        pathLens.append(len(pathList))
                        if len(pathList) <  m:
                            input[i][ii] = pathList + ["pad" for x in range(m-len(pathList))]

                accum = []
                for x in input:
                    accum += x

                tg = []
                for target in sent.targets:
                    tg += target

                # print(len(accum),len(tg))
                pred = model.predict(accum,pathLens,CUDA)
                assert(len(accum) == len(pred))
                
                all_predictions = all_predictions + pred
                all_gold = all_gold + tg
            # print(all_predictions)
            # print(all_gold)

            # print(len(all_predictions),len(all_gold))

            print(classification_report(all_gold,all_predictions))
            

 
    for epoch in range(args.epoch_num):


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for sent in tqdm(train.sents):
            if sent.targets == []:
                continue

            model.zero_grad()
            tg = []
            for target in sent.targets:
                tg += target

            target = convert_2_tensor(tg, tag_to_ix, torch.long, CUDA)
            # print(target)
            # print(sent.targets)
            #target = torch.stack(target,0)

            input = [[[z[0] for z in y] for y in x] for x in sent.samples]
            m=0
            for pathListList in input:
                for pathList in pathListList:
                    m = max(m, len(pathList))

            pathLens = []
            for i,pathListList in enumerate(input):
                for ii,pathList in enumerate(pathListList):
                    pathLens.append(len(pathList))
                    if len(pathList) <  m:
                        input[i][ii] = pathList + ["pad" for x in range(m-len(pathList))]

            accum = []
            for x in input:
                accum += x


            out, _ = model(input=accum,pathLens=pathLens,CUDA=CUDA)
            

            loss = loss_function(out, target)
            loss.backward(retain_graph = True)
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        if epoch % 1 != 0:
            optimizer.lr = optimizer.lr/2



        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)

    savePath = os.path.join(args.sys_path,f"{args.model_name}.pt")
    model.save(savePath)

    exit(0)

def run_train_paths(args):
    global CUDA

    test = json.load(open(os.path.join(args.sys_path,"path-data","test-path-data.json"),"r",encoding="utf-8"))
    test = PathData(test)

    train = json.load(open(os.path.join(args.sys_path,"path-data","train-path-data.json"),"r",encoding="utf-8"))
    train = PathData(train)

    all_tags = test.getTags() + train.getTags()

    tag_to_ix = {}
    dep_to_ix = {}

    deps = ['subtok','self','ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod', 'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp', 'compound', 'conj', 'csubj', 'csubjpass', 'dative', 'dep', 'det', 'dobj', 'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod', 'nsubj', 'nsubjpass', 'nummod', 'oprd', "parataxis", "pcomp", "pobj", "poss", "preconj", "predet", "prep", "prt", "punct", "quantmod", "relcl", "xcomp"]
    rdeps = ["r"+x for x in deps]

    all_deps = deps + rdeps + test.getDeps() + train.getDeps()

    for dep in all_deps:
        try:
            dep_to_ix[dep]
        except KeyError:
            dep_to_ix[dep] = len(dep_to_ix)

    for tag in all_tags:
        try:
            tag_to_ix[tag]
        except KeyError:
            tag_to_ix[tag] = len(tag_to_ix)


    pretrained_emb=None

    if CUDA:
        model = Elmo_paths(dep_to_ix,tag_to_ix,1024,1024,args.embeddings,args.sys_path).cuda()
    else:
        model = Elmo_paths(dep_to_ix,tag_to_ix,1024,1024,args.embeddings,args.sys_path)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)

    accum = []

    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal model 
        nonlocal accum
        nonlocal args


        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for sent in test.sents:
                if sent.targets == []:
                    continue

                all_predictions = all_predictions + model.predict(sent.sentence,sent.samples,CUDA)
                all_gold = all_gold + sent.targets
            #print(all_predictions)
            #print(all_gold)

            print(flat_classification_report(all_gold,all_predictions))
            

 
    for epoch in range(args.epoch_num):


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for sent in tqdm(train.sents):
            if sent.targets == []:
                continue

            model.zero_grad()

            target = [convert_2_tensor(x, tag_to_ix, torch.long, CUDA) for x in sent.targets]
            # print(target)
            # print(sent.targets)
            target = torch.stack(target,0)

            pred, out = model(input=sent.sentence,paths=sent.samples,CUDA=CUDA)
            out = torch.reshape(out,(out.shape[0],out.shape[2],out.shape[1]))

            # print(out.shape)
            # print(target.shape)

            loss = loss_function(out, target)
            loss.backward(retain_graph = True)
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        if epoch % 1 != 0:
            optimizer.lr = optimizer.lr/2



        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)

    savePath = os.path.join(args.sys_path,f"{args.model_name}.pt")
    model.save(savePath)

    exit(0)

def run_train_modifier(args):
    global CUDA

    test = json.load(open(os.path.join(args.sys_path,"data-fold{}".format(args.fold),"test-{}.json".format(args.data)),"r",encoding="utf-8"))
    test = [(x["span"],x["type"][0],) for x in test]

    train = json.load(open(os.path.join(args.sys_path,"data-fold{}".format(args.fold),"train-{}.json".format(args.data)),"r",encoding="utf-8"))
    train = [(x["span"],x["type"][0],) for x in train]

    print("Train size:",len(train))
    print("Test size:",len(test))


    all_data = test + train

    tag_to_ix = {}

    for sample in all_data:
        try:
            tag_to_ix[sample[1]]
        except KeyError:
            tag_to_ix[sample[1]] = len(tag_to_ix)

    def getBatches(data,size):
        tempBatch = {"spans":[],"lengths":[],"targets":[]}
        
        count = 0
        batches = []
        for x in train:
            if type(x[0]) != list:
                print(x[0])
            count+=1
            tempBatch["spans"].append(x[0])
            tempBatch["lengths"].append(len(x[0]))
            tempBatch["targets"].append(x[1])
            if count % size == 0:
                m = np.max(tempBatch["lengths"])
                tempBatch["spans"] = [span + ["<UNK>" for y in range(m-len(x))] for span in tempBatch["spans"]]
                batches.append(tempBatch)
                count = 0
                tempBatch = {"spans":[],"lengths":[],"targets":[]}

        m = np.max(tempBatch["lengths"])
        tempBatch["spans"] = [span+["<UNK>" for y in range(m-len(x))] for span in tempBatch["spans"]]
        batches.append(tempBatch)

        return batches
    batches = getBatches(train,128)
    #print(batches)

    pretrained_emb=None

    if CUDA:
        model = Modifier_Tagger(tag_to_ix,1024,1024,args.embeddings,args.sys_path).cuda()
    else:
        model = Modifier_Tagger(tag_to_ix,1024,1024,args.embeddings,args.sys_path)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    accum = []
    maxf1 = 0
    
    def check_test(test,epoch,avgLoss,fold,silent=True): 
        global CUDA
        nonlocal model 
        nonlocal accum
        nonlocal args
        nonlocal tag_to_ix
        nonlocal maxf1

        model.eval()


        with torch.no_grad():
            all_predictions = []
            all_gold = []
            for sample in test:
                pred = model.predict([sample[0]],CUDA)
                if pred == []:
                    continue
                all_predictions.append(pred[0][0])
                all_gold.append(sample[1])

            report = classification_report(all_gold,all_predictions)
            print(report)
            return getLabel(tag_to_ix,report)


    # model.load(os.path.join(args.sys_path,"modifier-pred.pt"))
    # check_test(test,epoch=1,avgLoss=1,fold=args.fold,silent=False)
    # exit(0)   

    
    for epoch in range(args.epoch_num):
        model.train()


        all_loss = list()

        print('Epoch: ' + str(epoch + 1))

        for i in tqdm(range(len(train))):
            model.zero_grad()

            target = train[i][1]
            target = convert_2_tensor([target], tag_to_ix, torch.long, CUDA)

            input = train[i][0]
            if input == []:
                continue
            out, pred = model(input, CUDA)

            loss = loss_function(out, target)
            loss.backward(retain_graph = True)
            optimizer.step()

            all_loss.append(loss.cpu().detach().numpy())

        # if epoch % 1 != 0:
        #     optimizer.lr = optimizer.lr/2



        print('\n')
        print('Average loss: ' + str(np.mean(all_loss)))

        print("Check Train:")
        #check_test(train,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)
        print("Check Dev:")
        f1 = check_test(test,epoch=(epoch+1),avgLoss=np.mean(all_loss),fold=args.fold,silent=False)
        print(f1)
         
        if f1 > maxf1:
            print("Saving model for new max weighted F1:{}".format(f1))
            maxf1 = f1
            savePath = os.path.join(args.sys_path,f"{args.model_name}fold{args.fold}.pt")
            model.save(savePath)

    

    exit(0)

def run_train_token(args):
    global CUDA

    print(args)

    test = load_samples(os.path.join(args.sys_path,"data","test_"+args.data+".tsv"))
    train = load_samples(os.path.join(args.sys_path,"data","train_"+args.data+".tsv"))
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
    subparser.add_argument("--type", "-t", default="modifier")

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






