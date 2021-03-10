import pandas
import sys
import os
import json
import pandas as pd
from spacy.tokens import Token
from src.model.pytorch_util import *
from src.model.load_util import *
from src.model.helpers import *
from src.model.data import *
from src.lib.helpers import *
from src.lib.evaluator import Evaluator

import src.graph as graph
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


    def evaluate(self,fold,complete=False,latex=False):
        ev = Evaluator(self.path,"prediction-tsv-partsys-fold{}".format(fold))
        ev.setupData()
        ev.testRegular(complete=complete,latex=False)
        ev.testClass(complete=complete,latex=latex)


    def predict(self,exerptController,CUDA,fold=1,match="old",usePathModel=False,complete=False,latex=False,model_name="",cp_name="",getother=None):
        """
        Helper methods at top
        """

        def getSpans(model,testData,data):
            spanDict = {}
            for x in testData:
                if x != "":

                    e = data[x]
                    test = [[x.text for x in sent] for sent in e.doc.sents]
                    predictions = model.predict(test,CUDA)
                    
                    count=0
                    for sent in predictions:
                        for pred in sent:
                            e.doc[count]._.prediction = pred
                            #print(e.doc[count].text,e.doc[count]._.prediction,e.doc[count]._.all)
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

                spanDict[x] = spans
            return spanDict
                
        def getGps(spans,pathModel=None):
            """
            Algorithm responsible for finding the related spans
            """
            def sentDist(span1,span2):
                doc = span1.doc
                s1 = span1.sent
                s2 = span2.sent
                i1,i2 = -10,500
                s = [x for x in doc.sents]
                for i, sent in enumerate(s):
                    if intersectSpanSpan(s1,sent):
                        i1=i

                    if intersectSpanSpan(s2,sent):
                        i2=i
                        
                return abs(i1-i2)

            def withinSentence(span1,span2):
                return sentDist(span1,span2) == 0

            def getRank(span1,span2):
                mul = sentDist(span1,span2) + 1
                shortestDist = min(abs(span1.start-span2.end),abs(span1.end-span2.start))
                return shortestDist #* mul


            def match(spans,tpe="ME",groups=None,exclusive=True):
                if groups == None:
                    groups = [{"QA":x} for x in spans["QA"]]

                qas = [x for x in spans["QA"]]
                mes = [x for x in spans[tpe]]
                
                if min(len(qas),len(mes)) == 0:
                    return groups
                
                ranked = []
                for i,qa in enumerate(qas):
                    for ii,me in enumerate(mes):
                        ranked.append((i,ii,getRank(qa[0],me)))

                ranked.sort(key=lambda x:x[2])
                # print(ranked)
                changed = 0

                matches = []
                i=0
                metaken=[]
                qataken=[]
                if exclusive == 1:
                    while(len(matches) < min(len(qas),len(mes)) or i < len(ranked)):
                        if(ranked[i][1] not in metaken and ranked[i][0] not in qataken):
                            matches.append(ranked[i])
                            metaken.append(ranked[i][1])
                            qataken.append(ranked[i][0])
                        i+=1
                elif exclusive == 2:
                    while(len(matches) < min(len(qas),len(mes)) or i < len(ranked)):
                        if(ranked[i][1] not in metaken and ranked[i][0] not in qataken):
                            if(withinSentence(qas[ranked[i][0]][0],mes[ranked[i][1]]) and ranked[i][2]<10):
                                matches.append(ranked[i])
                                metaken.append(ranked[i][1])
                                qataken.append(ranked[i][0])
                            else:
                                for idx,ii,r in ranked:
                                    if idx == ranked[i][0]:
                                        matches.append(ranked[idx])
                                        metaken.append(ranked[idx][1])
                                        qataken.append(ranked[idx][0])
                                        if ii != ranked[i][1]:
                                            print("changed",changed, tpe)
                                            changed +=1
                                        break
                        i+=1
                else:
                    while(len(matches) < len(qas) or i < len(ranked)):
                        if(ranked[i][0] not in qataken):
                            matches.append(ranked[i])
                            metaken.append(ranked[i][1])
                            qataken.append(ranked[i][0])
                        i+=1


                for i,ii,rank in matches:
                    if rank < 69 and sentDist(qas[i][0],mes[ii]) < 3 and tpe == "ME":
                        groups[i][tpe] = mes[ii]
                    elif rank < 27 and sentDist(qas[i][0],mes[ii]) < 2 and tpe == "MP":
                        groups[i][tpe] = mes[ii]
                    elif rank < 26 and sentDist(qas[i][0],mes[ii]) < 1 and tpe == "QL":
                        groups[i][tpe] = mes[ii]
                    else:
                        pass
                        # print(rank,sentDist(qas[i][0],mes[ii]),qas[i][0],mes[ii])

                return groups

            import pprint
            pp = pprint.PrettyPrinter(indent=4)

            exclusive = 2            
            groups = match(spans,tpe="ME",groups=None, exclusive=1)
            groups = match(spans,tpe="MP",groups=groups, exclusive=1)
            groups = match(spans,tpe="QL",groups=groups, exclusive=1)

            return groups


        def spanDist(s1,s2):
            return min(abs(s1.start_char-s2.end_char),abs(s2.start_char-s1.end_char))

        def getGroupings(spans,pathModel=None):
            """
            Algorithm responsible for finding the related spans
            """
            def withinSentence(span1,span2):
                doc = span1.doc
                s1 = span1.sent
                s2 = span2.sent
                i1,i2 = -10,500
                s = [x for x in doc.sents]
                for i, sent in enumerate(s):
                    if intersectSpanSpan(s1,sent):
                        i1=i

                    if intersectSpanSpan(s2,sent):
                        i2=i

                return abs(i1-i2) <= 1

            import pprint
            pp = pprint.PrettyPrinter(indent=4)

                
            groups = []
            if len(spans["QA"]) == 1:
                temp = {"QA":spans["QA"][0]}
                for x in ["ME","QL","MP"]:
                    if len(spans[x]) != 0:
                        for sp in spans[x]:
                            if withinSentence(temp["QA"][0],sp):
                                temp[x] = sp
                                break
                                  
                return [temp] 
            else:
                while(len(spans["QA"])>0):
                    temp = {"QA":spans["QA"].pop(0)}
                    for x in ["ME","QL","MP"]:
                        for i,z in enumerate(spans[x]):
                            if(intersectSpanSpan(temp["QA"][0].sent,z)):
                                temp[x] = spans[x].pop(i)
                                break
                    groups.append(temp)
                
                for x in ["ME","QL","MP"]:
                    if(len(spans[x]) > 0):
                        for z in spans[x]:
                            g = sorted([(spanDist(z,xx["QA"][0]),i)for i,xx in enumerate(groups)], key = lambda x : x[0], reverse=False)
                            for h in g:
                                try:
                                    groups[h[1]][x]
                                except KeyError:
                                    if withinSentence(groups[h[1]]["QA"][0],z):
                                        groups[h[1]][x] = z
                                        break

            def predictME(quant,pathModel):
                def getContext(span,window):
                    sent = span.sent
                    return span.doc[max(sent.start,span.i-window-1):min(span.i+window,sent.end- 1)]

                def getSource(span):
                    span = [tok for tok in span]
                    if span[-1].tag_ == "CD":
                        return span[-1]

                    for x in reversed(span): 
                        if x.tag_ in [ "CD","JJ"]:
                            return x

                    if len(span) == 1:
                        comp = span[-1]
                    else:
                        comp = span[-2]
                    print("in Graph class method getSource, compromise was made for {} with pos {} with quant {}".format(comp.text,comp.tag_," ".join([x.text for x in span])))
                    return comp

                doc = quant.doc
                headNouns = []

                for token in quant.sent:
                    if intersectSpanSpan(token,quant):
                        continue
                    
                    try: 

                        if token.tag_ in ["NNS","NN","NNP"]:
                            if intersectSpanSpan(doc[token.i+1],quant):
                                headNouns.append(token)
                            elif doc[token.i+1].tag_ in ["NNS","NN","NNP"]:
                                continue
                            else:
                                headNouns.append(token)

                    except IndexError:
                        continue
                         

                if headNouns == []:
                    return None
                
                a = graph.Graph(quant.sent)
                qa = getSource(quant)
                me = []
                for noun in headNouns:
                    if noun._.prediction in ["ME","MP","QA"]:
                        continue
                    path = a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(noun))
                    path = [x[0] for x in path][1:]
                    
                    if len(path) < 5:
                        try:
                            pred = pathModel.predictSys(
                                nounBatch=[x.text for x in getContext(noun,2)],
                                quantBatch=[x.text for x in quant],
                                pathList=path,
                                CUDA=CUDA)
                            if pred == ["ME"]:
                                me.append(noun)
                        except IndexError:
                            print("error predicting path for ",path)
                            continue

                    

                if me == []:
                    return None
                else: 
                    print("multiple options for ME found {}".format([("gold:",x._.all,"pred:",x._.prediction,x)for x in me]))
                    distanceRanking = [((qa.i-x.i)**2,x) for x in me].sort(key=lambda x:x[0])
                    print("distance rank {}".format(distanceRanking))
                    return doc[me[0].i-2:me[0].i]

            if pathModel != None:
                for x in groups:
                    try:
                        x["ME"]
                    except KeyError:
                        temp = predictME(x["QA"][0],pathModel)
                        
                        if temp == None:
                            continue
                        else:
                            x["ME"] = temp

            # pp.pprint([(x,[(y,y._.all) for y in x["QA"][0].sent]) for x in groups])
            # pp.pprint("spans",spans)
            return groups

        def getModifiers(spanDict,modifierModel):
            print("entering getModifiers(spanDict,modifierModel):")
            for key in spanDict.keys():
                try:
                    l = len(spanDict[key]["QA"])
                    if l ==0 :
                        continue
                    # print(type(spanDict[key]["QA"]))
                    a=[type(x)for x in spanDict[key]["QA"]]
                    #print(a)
                    if a[0] == type((1,2,3,)):
                        print(spanDict[key]["QA"])
                        continue
                    spanDict[key]["QA"] = [(quantSpan,modifierModel.predictSys([x.text for x in quantSpan],CUDA)) for quantSpan in spanDict[key]["QA"]]
                    assert(l == len(spanDict[key]["QA"]))
                except KeyError:
                    pass
            return spanDict


        def getOther(prediction,quantSpan,quantText):
            """
            Uses statistical prediction and a heuristic method to determine the class of the quantity
            and its unit
            """
            #prediction = model.predict([x.text for x in quantSpan],CUDA)[0][0]
            #prediction = model.predictSys([x.text for x in quantSpan],CUDA)
            if prediction == "IsCount":
                return {'mods': ['IsCount']}
            
            if(quantText.strip(" ")[-1].isnumeric()) and prediction == "NOMOD":
                return {}
            else: 
                for i in reversed(range(len(quantText))):
                    if str(quantText[i]).isnumeric() and prediction != "NOMOD":
                        unit = quantText[i+1:].strip(" ")
                        return {'mods': [prediction], 'unit':unit}
                    elif str(quantText[i]).isnumeric() and prediction == "NOMOD":
                        unit = quantText[i+1:].strip(" ")
                        return {'unit':unit}


            # return {}

            #below should be included            
            
            if prediction == "NOMOD":
                return {}
            else:
                return {'mods': [prediction]}


        def getOther_nomod(prediction,quantSpan,quantText):
            """
            No Modifier version of getOther
            """
            #prediction = model.predict([x.text for x in quantSpan],CUDA)[0][0]
            #prediction = model.predictSys([x.text for x in quantSpan],CUDA)
            if(quantText.strip(" ")[-1].isnumeric()):
                return {}
            else: 
                for i in reversed(range(len(quantText))):
                    if str(quantText[i]).isnumeric():
                        unit = quantText[i+1:].strip(" ")
                        return {'unit':unit}
            return {}

        def getOther_nounit(prediction,quantSpan,quantText):
            """
            No Unit Version of getOther
            """
            #prediction = model.predict([x.text for x in quantSpan],CUDA)[0][0]
            #prediction = model.predictSys([x.text for x in quantSpan],CUDA)
            if prediction == "NOMOD":
                return {}
            else: 
                return {'mods': [prediction]}

        


        def getOtherOrig(model,quantSpan,quantText):
            """
            A baseline prediction without using a statistical model
            """
            
            other = None
            if(quantText.strip(" ")[-1].isnumeric()):
                return {'mods': ['IsCount']}
            else: 
                for i in reversed(range(len(x["QA"].text))):
                    if str(quantText[i]).isnumeric():
                        return {'mods': ["IsRange"], 'unit':quantText[i+1:].strip(" ")}

            return {'mods': ['IsCount']}


        if not self.ready:
            print("Error in System predict(), the current system is not ready\
             for this task. Please load the system correctly")

        if(os.path.isfile(os.path.join(self.path,"data-fold{}".format(fold),"test.txt"))):
            self.testData = open(os.path.join(self.path,"data-fold{}".format(fold),"test.txt"),"r",encoding="utf-8").read().split("\n")
        else:
            print("Error in System load(), the passed path were invalid.\
             Please load the system correctly")
            self.testData = None
            return False
             
        gazetteer = open(os.path.join(self.path,"gazetteers/combined_measurements.lst"),"r",encoding="utf-8").read().split("\n")
        gazetteer = {x.lower():1 for x in gazetteer}

        if "scibert" in model_name:
            model_name = 'allenai/scibert_scivocab_cased'
        elif "base" in model_name:
            model_name = 'bert-base-cased'
        elif "roberta" in model_name:
            model_name = 'roberta-base'
        elif "bart" in model_name:
            model_name = 'facebook/bart-base'
        elif "biobert" in model_name:
            model_name = 'dmis-lab/biobert-base-cased-v1.1'
        elif "bioroberta" in model_name:
            model_name = 'allenai/biomed_roberta_base'



        if getother == "nounit":
            getOther = getOther_nounit
        elif getother == "nomod":
            getOther = getOther_nomod

        
        #Load Bert Model
        print("Loading bert for token classificaiton...")

        temp = torch.load(os.path.join(self.path,"{}-fold{}.pt".format(cp_name,fold)))
        tag_to_ix = temp["tag_to_ix"]
        del temp

        if model_name in ['dmis-lab/biobert-base-cased-v1.1','facebook/bart-base','roberta-base','allenai/biomed_roberta_base']:
            if CUDA:
                model = RobertaTokenClassifier(model_name,tag_to_ix).cuda()
            else:
                model = RobertaTokenClassifier(model_name,tag_to_ix)

        else:

            if CUDA:
                model = BERT_SequenceTagger(model_name,len(tag_to_ix),tag_to_ix).cuda()
            else:
                model = BERT_SequenceTagger(model_name,len(tag_to_ix),tag_to_ix)
 
        model.load(os.path.join(self.path,"{}-fold{}.pt".format(cp_name,fold)))
        # END BERT LOAD

        try:
            exerptController[self.testData[0]].doc[0]._.prediction
        except Exception: 
            Token.set_extension("prediction", default="o", force=True)

        print("Predicting spans...")
        spanDict = getSpans(model,self.testData,exerptController.data)

        del model
        

        #LOAD MODIFIER BERT
        temp = torch.load(os.path.join(self.path,"bert-mod-fold{}.pt".format(fold)))
        tag_to_ix = temp["tag_to_ix"]
        del temp


        model_name = 'allenai/scibert_scivocab_cased'
        print("Loading bert for modifier classificaiton...")

        if CUDA:
            model_ = BERT_Matcher( model_name, len(tag_to_ix),tag_to_ix).cuda()
        else:
            model_ = BERT_Matcher( model_name, len(tag_to_ix),tag_to_ix)

        model_.load(os.path.join(self.path,"bert-mod-fold{}.pt".format(fold)))

        print("Predicting modifiers...")
        spanDict = getModifiers(spanDict,model_)
        del model_

        #END BERT MODIFIER LOAD

        
        if usePathModel == False:
            print("No path Model")
            pathModel = None
        

        # for x in spanDict.keys():

        #     length = len(spanDict[x]["QL"])
        #     for i, span in enumerate(reversed(spanDict[x]["QL"])):
        #         tempPred = pruner.predict([[x.text for x in span]],1)
        #         if tempPred != [["QL"]]:
        #             print(span,tempPred)
        #             spanDict[x]["QL"].pop(length-i-1)



        filecount = 0
        for docId, spans in spanDict.items():
            try:
                e = exerptController.data[docId]
            except KeyError: 
                print(docId,spans)

            mapping = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
            df = None 
            annotSet = 1
            tid = 1 
            if match == "old":
                groups = getGroupings(spans,pathModel)
            else:
                groups = getGps(spans,pathModel) 

            for x in groups:
                other = getOther(prediction=x["QA"][1],quantSpan=x["QA"][0],quantText=x["QA"][0].text)
                #print("other:{} for quantity: {}".format(other,x["QA"][0].text))
                #other = getOtherOrig(model_,x["QA"],x["QA"].text)

                row = {
                    "docId":[e.name],
                    "annotSet": [annotSet],
                    "annotType": "Quantity",
                    "startOffset": [x["QA"][0].start_char],
                    "endOffset": [x["QA"][0].end_char], 
                    "annotId": [f"T{tid}"],
                    "text":[x["QA"][0].text],
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

            if not os.path.isdir(os.path.join(self.path,"prediction-tsv-partsys-fold{}".format(fold))):
                os.mkdir(os.path.join(self.path, "prediction-tsv-partsys-fold{}".format(fold)))

            def func(x):
                x[7] = json.dumps(x[7])
                return x

            filecount+=1
            if type(df) != type(None):
                df = df.apply(lambda x : func(x),axis=1)
                df.to_csv(open(os.path.join(self.path, "prediction-tsv-partsys-fold{}".format(fold),f"{e.name}.tsv"),"w",encoding="utf-8"),sep = "\t",header=True, index = False)
                #print("Creating TSV File #{} at {}".format(filecount,os.path.join(self.path, "prediction-tsv-onesys-fold{}".format(fold),f"{e.name}.tsv")))
            else:
                pass
                # print("df is {} for docid {}".format(df,e.name))
                # print(spans)  

        ev = Evaluator(self.path,"prediction-tsv-partsys-fold{}".format(fold))

        if fold == 1001:
            ev.setupData(eval=True)
        else:
            ev.setupData()

        # ev.testRegular(complete=complete,latex=False)
        ev.testClass(complete=complete,latex=latex)

                            
    def predict2(self,exerptController,CUDA,fold=1):
            """
            Helper methods at top
            """

            def getSpans(model,testData,data):
                spanDict = {}
                for x in testData:
                    if x != "":

                        e = data[x]
                        sents = [s for s in e.doc.sents]
                        count = 0
                        for sent in e.doc.sents:
                            #print([(tok.text,tok._.all) for tok in sent])
                            tokenPred, modPred = model.predict([tok.text if tok.text != ' ' else 'the' for tok in sent],True)
                            qaCount = 0
                            inspan=False
                            for pred in tokenPred:
                                e.doc[count]._.prediction = pred
                                if pred=="QA" and inspan == False:
                                    inspan = True
                                    if len(modPred[qaCount]) <= 2:
                                        e.doc[count]._.qamodifier = modPred[qaCount]
                                    else:
                                        print(modPred[qaCount])
                                        e.doc[count]._.prediction = "o"
                                elif pred=="QA" and inspan == True:
                                    if len(modPred[qaCount]) <= 2:
                                        e.doc[count]._.qamodifier = modPred[qaCount]
                                    else:
                                        print(modPred[qaCount])
                                        e.doc[count]._.prediction = "o"
                                elif pred == "o" and inspan == True:
                                    qaCount +=1
                                    inspan = False
                                count+=1


                        # test = [[x.text for x in sent] for sent in e.doc.sents]
                        # predictions = model.predict(test,CUDA)
                        
                        # count=0
                        # for sent in predictions:
                        #     for pred in sent:
                        #         e.doc[count]._.prediction = pred
                        #         #print(e.doc[count].text,e.doc[count]._.prediction,e.doc[count]._.all)
                        #         count+=1

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

                    spanDict[x] = spans
                return spanDict
                    
                    

            def spanDist(s1,s2):
                return min(abs(s1.start_char-s2.end_char),abs(s2.start_char-s1.end_char))

            def getGroupings(spans,pathModel=None):
                """
                Algorithm responsible for finding the related spans
                """
                def withinSentence(span1,span2):
                    doc = span1.doc
                    s1 = span1.sent
                    s2 = span2.sent
                    i1,i2 = -10,500
                    s = [x for x in doc.sents]
                    for i, sent in enumerate(s):
                        if intersectSpanSpan(s1,sent):
                            i1=i

                        if intersectSpanSpan(s2,sent):
                            i2=i

                    return abs(i1-i2) <= 1

                import pprint
                pp = pprint.PrettyPrinter(indent=4)
                
                

                    
                groups = []
                if len(spans["QA"]) == 1:
                    temp = {"QA":spans["QA"][0]}
                    for x in ["ME","QL","MP"]:
                        if len(spans[x]) != 0:
                            for sp in spans[x]:
                                if withinSentence(temp["QA"],sp):
                                    temp[x] = sp
                                    break
                                    
                    print("groups:\n")
                    pp.pprint([(x,[(y,y._.all) for y in x["QA"][0].sent]) for x in groups])
                    pp.pprint("spans",spans)
                    return [temp] 
                else:
                    while(len(spans["QA"])>0):
                        temp = {"QA":spans["QA"].pop(0)}
                        for x in ["ME","QL","MP"]:
                            for i,z in enumerate(spans[x]):
                                if(intersectSpanSpan(temp["QA"][0].sent,z)):
                                    temp[x] = spans[x].pop(i)
                                    break
                        groups.append(temp)
                    
                    for x in ["ME","QL","MP"]:
                        if(len(spans[x]) > 0):
                            for z in spans[x]:
                                g = sorted([(spanDist(z,xx["QA"]),i)for i,xx in enumerate(groups)], key = lambda x : x[0], reverse=False)
                                for h in g:
                                    try:
                                        groups[h[1]][x]
                                    except KeyError:
                                        if withinSentence(groups[h[1]]["QA"],z):
                                            groups[h[1]][x] = z
                                            break

                def predictME(quant,pathModel):
                    def getSource(span):
                        span = [tok for tok in span]
                        if span[-1].tag_ == "CD":
                            return span[-1]

                        for x in reversed(span): 
                            if x.tag_ in [ "CD","JJ"]:
                                return x

                        if len(span) == 1:
                            comp = span[-1]
                        else:
                            comp = span[-2]
                        print("in Graph class method getSource, compromise was made for {} with pos {} with quant {}".format(comp.text,comp.tag_," ".join([x.text for x in span])))
                        return comp

                    doc = quant.doc
                    headNouns = []

                    for token in quant.sent:
                        if intersectSpanSpan(token,quant):
                            continue
                        
                        try: 

                            if token.tag_ in ["NNS","NN","NNP"]:
                                if intersectSpanSpan(doc[token.i+1],quant):
                                    headNouns.append(token)
                                elif doc[token.i+1].tag_ in ["NNS","NN","NNP"]:
                                    continue
                                else:
                                    headNouns.append(token)

                        except IndexError:
                            continue
                            

                    if headNouns == []:
                        return None
                    
                    a = graph.Graph(quant.sent)
                    qa = getSource(quant)
                    me = []
                    for noun in headNouns:
                        if noun._.prediction in ["ME","MP","QA"]:
                            continue
                        path = a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(noun))
                        path = [x[0] for x in path][1:]

                        try:
                            pred = pathModel.predictSys(path,True)
                        except IndexError:
                            print("error predicting path for ",path)
                            continue

                        if pred == ["ME"]:
                            me.append(noun)
                        

                    
                    
                    print("multiple options for ME found {}".format([(x._.all,x._.prediction,x)for x in me]))

                    if me == []:
                        return None
                    else: 
                        return doc[me[0].i-2:me[0].i]

                if pathModel != None:
                    for x in groups:
                        try:
                            x["ME"]
                        except KeyError:
                            temp = predictME(x["QA"],pathModel)
                            
                            if temp == None:
                                continue
                            else:
                                x["ME"] = temp

                
                print("groups:\n")
                pp.pprint([(x,[(y,y._.all) for y in x["QA"][0].sent]) for x in groups])
                pp.pprint("spans",spans)
                return groups


            def getOther(model,quantSpan,quantText):
                """
                Uses statistical prediction and a heuristic method to determine the class of the quantity
                and its unit
                """
                #prediction = model.predict([x.text for x in quantSpan],CUDA)[0][0]
                prediction = [x for x in quantSpan][0]._.qamodifier
                if "IsCount" in prediction:
                    if "Nomod" in prediction:
                        return {'mods': ["isCount"]}
                    else:
                        return {'mods': prediction}
                
                if(quantText.strip(" ")[-1].isnumeric()) and "Nomod" in prediction:
                    return {}
                else: 
                    for i in reversed(range(len(x["QA"].text))):
                        if str(quantText[i]).isnumeric() and "Nomod" not in prediction:
                            unit = quantText[i+1:].strip(" ")
                            return {'mods': prediction, 'unit':unit}
                        elif str(quantText[i]).isnumeric() and "Nomod" in prediction:
                            unit = quantText[i+1:].strip(" ")
                            return {'unit':unit}

                return {}

            def getOtherOrig(model,quantSpan,quantText):
                """
                A baseline prediction without using a statistical model
                """
                
                other = None
                if(quantText.strip(" ")[-1].isnumeric()):
                    return {'mods': ['IsCount']}
                else: 
                    for i in reversed(range(len(x["QA"].text))):
                        if str(quantText[i]).isnumeric():
                            return {'mods': ["IsRange"], 'unit':quantText[i+1:].strip(" ")}

                return {'mods': ['IsCount']}


            if not self.ready:
                print("Error in System predict(), the current system is not ready\
                for this task. Please load the system correctly")

            if(os.path.isfile(os.path.join(self.path,"data-fold{}".format(fold),"test.txt"))):
                self.testData = open(os.path.join(self.path,"data-fold{}".format(fold),"test.txt"),"r",encoding="utf-8").read().split("\n")
            else:
                print("Error in System load(), the passed path were invalid.\
                Please load the system correctly")
                self.testData = None
                return False
                
            gazetteer = open(os.path.join(self.path,"gazetteers/combined_measurements.lst"),"r",encoding="utf-8").read().split("\n")
            gazetteer = {x.lower():1 for x in gazetteer}

            
            #Load Bert Model

            model_name = 'allenai/scibert_scivocab_cased'
            loaded=torch.load(f'{self.path}/bert_trained_model-modif-loss-fold{fold}.pt')

            if CUDA:
                model = BERT_SequenceTagger2(model_name, loaded['tag_to_ix'], loaded['modif_to_ix']).cuda()
            else:
                model = BERT_SequenceTagger2(model_name, loaded['tag_to_ix'], loaded['modif_to_ix'])

            model.load_state_dict(loaded['state_dict'])

            #end load bert model

            try:
                exerptController[self.testData[0]].doc[0]._.prediction
            except Exception: 
                Token.set_extension("prediction", default="o", force=True)

            try:
                exerptController[self.testData[0]].doc[0]._.qamodifier
            except Exception: 
                Token.set_extension("qamodifier", default=[], force=True)

            spanDict = getSpans(model,self.testData,exerptController.data)
            
            filecount = 0
            for docId, spans in spanDict.items():
                try:
                    e = exerptController.data[docId]
                except KeyError: 
                    print(docId,spans)
                mapping = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
                df = None 
                annotSet = 1
                tid = 1 
                for x in getGroupings(spans):
                    other = getOther(model,x["QA"],x["QA"].text)
                    #other = getOtherOrig(model_,x["QA"],x["QA"].text)

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

                if not os.path.isdir(os.path.join(self.path,"prediction-tsv-onesys-fold{}".format(fold))):
                    os.mkdir(os.path.join(self.path, "prediction-tsv-onesys-fold{}".format(fold)))

                def func(x):
                    x[7] = json.dumps(x[7])
                    return x

                filecount += 1
                if type(df) != type(None):
                    df = df.apply(lambda x : func(x),axis=1)
                    df.to_csv(open(os.path.join(self.path, "prediction-tsv-onesys-fold{}".format(fold),f"{e.name}.tsv"),"w",encoding="utf-8"),sep = "\t",header=True, index = False)
                    print("Creating TSV File #{} at {}".format(filecount,os.path.join(self.path, "prediction-tsv-onesys-fold{}".format(fold),f"{e.name}.tsv")))      

