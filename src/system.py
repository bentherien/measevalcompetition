import pandas
import sys
import os
import json
import pandas as pd
from spacy.tokens import Token
from src.model.pytorch_util import Sequence_Tagger,Sequence_Tagger_GCNN, BERT_SequenceTagger, Modifier_Tagger
from src.model.load_util import *
from src.model.helpers import *
from src.lib.helpers import *


class System:
    def __init__(self, pathToSystem, modelname):
        self.path = pathToSystem
        self.ready = self.load(modelname)
        
    def load(self, modelname):
        if(os.path.isfile(os.path.join(self.path,"test.txt"))):
            self.testData = open(os.path.join(self.path,"test.txt"),"r",encoding="utf-8").read().split("\n")
        else:
            print("Error in System load(), the passed path were invalid.\
             Please load the system correctly")
            self.testData = None
            return False

        if(os.path.isfile(os.path.join(self.path,modelname))):
            self.modelname = modelname
        else:
            print("Error in System load(), the passed path were invalid.\
             Please load the system correctly")
            self.modelname = None
            return False

        return True


    def predict(self,exerptController,CUDA):
        if not self.ready:
            print("Error in System predict(), the current system is not ready\
             for this task. Please load the system correctly")
             
        gazetteer = open(os.path.join(self.path,"gazetteers/combined_measurements.lst"),"r",encoding="utf-8").read().split("\n")
        gazetteer = {x.lower():1 for x in gazetteer}

        test = load_samples(os.path.join(self.path,"data","test_allS.tsv"))
        train = load_samples(os.path.join(self.path,"data","train_allS.tsv"))
        all_data = test + train

        pretrained_emb = None
        emb = "pubmed"

        word_to_ix = {}
        tag_to_ix = {}
        pos_to_ix={}

        for sample in all_data:
            for token in sample.tokens:
                if token not in word_to_ix:
                    word_to_ix[token] = len(word_to_ix)

            for tag in sample.targets:
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)

            for pos in sample.pos:
                if pos not in pos_to_ix:
                    pos_to_ix[pos] = len(pos_to_ix)
        
        """
        if CUDA:
            model = Sequence_Tagger(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,emb,self.path).cuda()
        else:
            model = Sequence_Tagger(word_to_ix,tag_to_ix,pos_to_ix,pretrained_emb,1024,1024,emb,self.path)
        """

        

        if CUDA:
            model = BERT_SequenceTagger( 'allenai/scibert_scivocab_cased', len(tag_to_ix),tag_to_ix).cuda()
        else:
            model = BERT_SequenceTagger( 'allenai/scibert_scivocab_cased', len(tag_to_ix),tag_to_ix)


        print(model.tag_to_ix)
        
        model.load(os.path.join(self.path,self.modelname))

        print(model.tag_to_ix)

        
        #load model for quant class
        all_data = json.load(open(os.path.join(self.path,"data-json","mod-data.json"),"r",encoding="utf-8"))

        all_data = [(x["span"],x["type"][0],) for x in all_data]

        tag_to_ix = {}

        for sample in all_data:
            try:
                tag_to_ix[sample[1]]
            except KeyError:
                tag_to_ix[sample[1]] = len(tag_to_ix)

        if CUDA:
            model_ = Modifier_Tagger(tag_to_ix,1024,1024,"pubmed",self.path).cuda()
        else:
            model_ = Modifier_Tagger(tag_to_ix,1024,1024,"pubmed",self.path)

        model_.load(os.path.join(self.path,"modifier-pred.pt"))

        

        try:
            exerptController[self.testData[0]].doc[0]._.prediction
        except Exception: 
            Token.set_extension("prediction", default="o", force=True)

        for x in self.testData:
            if x != "":

                e = exerptController.data[x]
                test = [[x.text for x in sent] for sent in e.doc.sents]
                predictions = model.predict(test,CUDA)


                def fillInPredictions(predictions):
                    count = 0
                    inspan = False
                    currentSpan = ""
                    spanI = -1
                    lastI = -1
                    
                    for sent in predictions:
                        inspan = False
                        spanI = -1
                        lastI = -1
                        for pred in sent:
                            e.doc[count]._.prediction = pred

                            if pred != "o" and inspan == False:
                                inspan = True
                                currentSpan = pred
                                spanI = count
                                lastI = -1
                            elif pred == "o":
                                continue
                            elif pred != "o" and pred != currentSpan:
                                if lastI != -1:
                                        for i in range(spanI+1,lastI+1):
                                            e.doc[i]._.prediction = currentSpan
                                inspan = False
                                currentSpan = ""
                                spanI = -1
                                lastI = -1

                            elif pred == "ME" and currentSpan == "ME"  and inspan:
                                pass
                            elif pred == "MP" and currentSpan == "MP"  and inspan:
                                pass
                            elif pred == "QA" and currentSpan == "QA"  and inspan:
                                pass
                            elif pred == "QL" and currentSpan == "QL"  and inspan:
                                pass

                            elif pred != "o" and inspan == True and pred == currentSpan:\
                                #extend spans for no more than 4
                                if count - spanI  < 4:
                                    lastI = count
                                else:
                                    if lastI != -1:
                                        for i in range(spanI+1,lastI+1):
                                            e.doc[i]._.prediction = currentSpan
                                    lastI = -1 
                                    spanI = count
                            elif pred != "o" and inspan == True and pred != currentSpan:  
                                if(spanI != -1):
                                    if lastI != -1:
                                        for i in range(spanI+1,lastI+1):
                                            e.doc[i]._.prediction = currentSpan
                                currentSpan = pred
                                spanI = count
                                lastI = -1
                            count+=1

                        if spanI != -1 and lastI != -1:
                            for i in range(spanI+1,lastI+1):
                                            e.doc[i]._.prediction = currentSpan
                
                count=0
                for sent in predictions:
                    for pred in sent:
                        e.doc[count]._.prediction = pred
                        count+=1

                spans = {"ME":[],"MP":[],"QL":[],"QA":[]}
                inSpan = False
                spanTpe = ""
                start = -1
                end = -1
                count = -1 
                for sent in e.doc.sents:
                    if inSpan == True:
                        if end == -1:
                            spans[spanTpe].append(e.doc[start:start+1])
                        else:
                            spans[spanTpe].append(e.doc[start:end+1])
                        inSpan = False
                        start = -1
                        end = -1
                        spanTpe = ""
                    else:
                        start = -1
                        end = -1
                        spanTpe = ""

                    for token in sent:
                        count+=1
                        if token._.prediction == "o":
                            if inSpan == True:
                                if end == -1:
                                    spans[spanTpe].append(e.doc[start:start+1])
                                else:
                                    spans[spanTpe].append(e.doc[start:end+1])
                                inSpan = False
                                start = -1
                                end = -1
                                spanTpe = ""
                            else:
                                continue
                        elif token._.prediction == "ME":
                            if inSpan == True and spanTpe != "ME":
                                if end == -1:
                                    spans[spanTpe].append(e.doc[start:start+1])
                                else:
                                    spans[spanTpe].append(e.doc[start:end+1])
                                start = count
                                end = -1
                                spanTpe = token._.prediction
                            elif inSpan == True and spanTpe == "ME":
                                end = count
                            else:
                                inSpan = True
                                start = count
                                spanTpe = token._.prediction
                        elif token._.prediction == "MP":
                            if inSpan == True and spanTpe != "MP":
                                if end == -1:
                                    spans[spanTpe].append(e.doc[start:start+1])
                                else:
                                    spans[spanTpe].append(e.doc[start:end+1])
                                start = count
                                end = -1
                                spanTpe = token._.prediction
                            elif inSpan == True and spanTpe == "MP":
                                end = count
                            else:
                                inSpan = True
                                start = count
                                spanTpe = token._.prediction
                        elif token._.prediction == "QA":
                            if inSpan == True and spanTpe != "QA":
                                if end == -1:
                                    spans[spanTpe].append(e.doc[start:start+1])
                                else:
                                    spans[spanTpe].append(e.doc[start:end+1])
                                start = count
                                end = -1
                                spanTpe = token._.prediction
                            elif inSpan == True and spanTpe == "QA":
                                end = count
                            else:
                                inSpan = True
                                start = count
                                spanTpe = token._.prediction
                        elif token._.prediction == "QL":
                            if inSpan == True and spanTpe != "QL":
                                if end == -1:
                                    spans[spanTpe].append(e.doc[start:start+1])
                                else:
                                    spans[spanTpe].append(e.doc[start:end+1])
                                start = count
                                end = -1
                                spanTpe = token._.prediction
                            elif inSpan == True and spanTpe == "QL":
                                end = count
                            else:
                                inSpan = True
                                start = count
                                spanTpe = token._.prediction
                
                

                def spanDist(s1,s2):
                    return min(abs(s1.start_char-s2.end_char),abs(s2.start_char-s1.end_char))

                def getGroupings(spans):
                    groups = []
                    if len(spans["QA"]) == 1:
                        temp = {"QA":spans["QA"][0]}
                        for x in ["QL","ME","MP"]:
                            if len(spans[x]) != 0:
                                temp[x] = spans[x][0]
                        return [temp] 
                    else:
                        while(len(spans["QA"])>0):
                            temp = {"QA":spans["QA"].pop(0)}
                            for x in ["QL","ME","MP"]:
                                for i,z in enumerate(spans[x]):
                                    if(intersectSpanSpan(temp["QA"][0].sent,z)):
                                        temp[x] = spans[x].pop(i)
                                        break
                            groups.append(temp)
                        
                        for x in ["QL","ME","MP"]:
                            if(len(spans[x]) > 0):
                                for z in spans[x]:
                                    g = sorted([(spanDist(z,xx["QA"]),i)for i,xx in enumerate(groups)], key = lambda x : x[0], reverse=False)
                                    for h in g:
                                        try:
                                            groups[h[1]][x]
                                        except KeyError:
                                            groups[h[1]][x] = z
                                            break
                    return groups


                def getOther(model,quantSpan,quantText):
                    """
                    Uses statistical prediction and a heuristic method to determine the class of the quantity
                    and its unit
                    """
                    prediction = model.predict([x.text for x in quantSpan],CUDA)[0][0]
                    if prediction == "IsCount":
                        return {'mods': ['IsCount']}
                    
                    other = None
                    if(quantText.strip(" ")[-1].isnumeric()):
                        return {'mods': ['IsCount']}
                    else: 
                        for i in reversed(range(len(x["QA"].text))):
                            if str(quantText[i]).isnumeric():
                                return {'mods': [prediction], 'unit':quantText[i+1:]}
                                

                    if other == None:
                        return {'mods': ['IsCount']}


                mapping = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
                df = None 
                annotSet = 1
                tid = 1 
                for x in getGroupings(spans):
                    other = getOther(model_,x["QA"],x["QA"].text)

                    row = {
                        "docId":[e.name],
                        "annotSet": [annotSet],
                        "annotType": "Quantity",
                        "startOffset": [x["QA"].start_char],
                        "endOffset": [x["QA"].end_char], 
                        "annotId": [f"T{tid}"],
                        "text":[x["QA"].text],
                        "other": [other]
                        }

                    if type(df) == type(None):
                        df = pd.DataFrame.from_dict(row, orient = "columns")
                    else:
                        df1 = pd.DataFrame.from_dict(row, orient = "columns")
                        df = df.append(df1)
                    quantId = tid
                    tid += 1


                    if "MP" in x:
                        row = {
                            "docId":[e.name],
                            "annotSet": [annotSet],
                            "annotType": "MeasuredProperty",
                            "startOffset": [x["MP"].start_char],
                            "endOffset": [x["MP"].end_char], 
                            "annotId": [f"T{tid}"],
                            "text":[x["MP"].text],
                            "other": [{"HasQuantity": f"T{quantId}"}]
                        }
                        df1 = pd.DataFrame.from_dict(row, orient = "columns")
                        df = df.append(df1)
                        tid += 1

                        if "ME" in x:
                            row = {
                                "docId":[e.name],
                                "annotSet": [annotSet],
                                "annotType": "MeasuredEntity",
                                "startOffset": [x["ME"].start_char],
                                "endOffset": [x["ME"].end_char], 
                                "annotId": [f"T{tid}"],
                                "text":[x["ME"].text],
                                "other": [{"HasProperty": "T{}".format(tid-1)}]
                            }
                            df1 = pd.DataFrame.from_dict(row, orient = "columns")
                            df = df.append(df1)
                            tid += 1
                    else:
                        if "ME" in x:
                            row = {
                                "docId":[e.name],
                                "annotSet": [annotSet],
                                "annotType": "MeasuredEntity",
                                "startOffset": [x["ME"].start_char],
                                "endOffset": [x["ME"].end_char], 
                                "annotId": [f"T{tid}"],
                                "text":[x["ME"].text],
                                "other": [{"HasQuantity": f"T{quantId}"}]
                            }
                            df1 = pd.DataFrame.from_dict(row, orient = "columns")
                            df = df.append(df1)
                            tid += 1

                    if "QL" in x:
                        row = {
                            "docId":[e.name],
                            "annotSet": [annotSet],
                            "annotType": "Qualifier",
                            "startOffset": [x["QL"].start_char],
                            "endOffset": [x["QL"].end_char], 
                            "annotId": [f"T{tid}"],
                            "text":[x["QL"].text],
                            "other": [{"Qualifies": f"T{quantId}"}]
                        }
                        df1 = pd.DataFrame.from_dict(row, orient = "columns")
                        df = df.append(df1)
                        tid += 1

                    annotSet +=1

                if not os.path.isdir(os.path.join(self.path, "prediction-tsv")):
                    os.mkdir(os.path.join(self.path, "prediction-tsv"))

                def func(x):
                    x[7] = json.dumps(x[7])
                    return x

                if type(df) != type(None):
                    df = df.apply(lambda x : func(x),axis=1)
                    df.to_csv(open(os.path.join(self.path, "prediction-tsv",f"{e.name}.tsv"),"w",encoding="utf-8"),sep = "\t",header=True, index = False)

                            
        


""" Before stuff
e.doc[count]._.prediction = pred

                        for sent in predictions:
                    inspan = False
                    spanI = -1
                    lastI = -1
                    for pred in sent:
                        e.doc[count]._.prediction = pred

                        if pred[:2] == "B-" and inspan == False:
                            inspan = True
                            currentSpan = pred[2:]
                            spanI = count
                            lastI = -1
                        elif pred[:2] == "B-" and inspan == True:
                            if(spanI != -1):
                                for i in range(spanI+1,lastI+1):
                                    e.doc[i]._.prediction = "I-"+currentSpan
                            inspan = True
                            currentSpan = pred[2:]
                            spanI = count
                            lastI = -1
                        elif pred[:2] == "I-" and inspan == True and currentSpan == pred[2:]:
                            lastI = count
                        count+=1
"""

        