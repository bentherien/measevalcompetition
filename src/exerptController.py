import logging
import time
import random
import math
import os
import json
import spacy
import pandas as pd
import copy
import numpy as np

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from spacy.tokens import Token
from spacy.tokens import Span
from spacy.tokens import Doc

import src.graph as graph
from src.lib.helpers import *
from src.exerpt import Exerpt
from src.lib.helpers import createAnnotation, createFeature, createNode


class ExerptController:
    """
    Description: Class containing all exerpts that encapsulates all the logic needed to deal with them 
    """

    def __init__(self, filepath, evaluate = False):
        if evaluate == False:
            if(os.path.isdir(filepath)):
                self.data = self.readData(filepath)
            else:
                self.data = {}

            self.annotateData()
            self.annotateRelations()
            self.getData()
        else:
            if(os.path.isdir(filepath)):
                self.data = self.readDataEval(filepath)
            else:
                self.data = {}


    def __iter__(self):
        return iter(self.data.values())

    def initGraphs(self):
        for x in self.data.values():
            x.initGraphs()

    def annotateData(self):
        Token.set_extension("all", default="o", force=True)
        Token.set_extension("quant", default="o", force=True)
        Token.set_extension("qual", default="o", force=True)
        Token.set_extension("me", default="o", force=True)
        Token.set_extension("mp", default="o", force=True)
        Token.set_extension("other", default="o", force=True)
        Token.set_extension("annotId", default=-1, force=True)
        tempDict = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
        for x in self.data.values():
            for annot in x.doc._.meAnnots.values():
                for key in annot.keys():
                    if key != "sentences":
                        tempSpanlen = len(annot[key])
                        for token in annot[key]["span"]:
                            token._.annotId = annot[key]["other"]["annotID"]
                            if type(token._.other) == dict:
                                token._.other["annotID"].append(annot[key]["other"]["annotID"])
                            else:
                                token._.other = copy.deepcopy(annot[key]["other"])
                                token._.other["annotID"] = [token._.other["annotID"]]
                                

                            token._.all = tempDict[key]
                            if key == "MeasuredEntity":
                                token._.me = tempDict[key]
                            elif key == "Quantity":
                                token._.quant = tempDict[key]
                            elif key == "Qualifier":
                                token._.qual = tempDict[key]
                            elif key == "MeasuredProperty":
                                token._.mp = tempDict[key]


    def annotateRelations(self):
        Token.set_extension("relationDoc", default=(0,"root",0), force=True)
        Token.set_extension("relationSent", default=(0,"root",0), force=True)
        Doc.set_extension("ME", default=[], force=True)
        Doc.set_extension("MP", default=[], force=True)
        Doc.set_extension("QL", default=[], force=True)
        Doc.set_extension("QA", default=[], force=True)
        Doc.set_extension("qa_me_rel", default=[], force=True)
        Doc.set_extension("qa_mp_rel", default=[], force=True)
        Doc.set_extension("mp_me_rel", default=[], force=True)
        Doc.set_extension("qa_ql_rel", default=[], force=True)
        Doc.set_extension("modifiers", default=[], force=True)

        Span.set_extension("ME", default=[], force=True)
        Span.set_extension("MP", default=[], force=True)
        Span.set_extension("QL", default=[], force=True)
        Span.set_extension("QA", default=[], force=True)
        Span.set_extension("qa_me_rel", default=[], force=True)
        Span.set_extension("qa_mp_rel", default=[], force=True)
        Span.set_extension("mp_me_rel", default=[], force=True)
        Span.set_extension("qa_ql_rel", default=[], force=True)
        Span.set_extension("modifiers", default=[], force=True)


        tempDict = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
        for x in self.data.values():
            e = x
            for annot in x.doc._.meAnnots.values():

                #at the document level
                spanQuant = annot["Quantity"]["span"].start
                e.doc._.QA.append((annot["Quantity"]["span"].start,annot["Quantity"]["span"].end- 1,))
                try:
                    e.doc._.modifiers.append(annot["Quantity"]["other"]["mods"])
                except KeyError:
                    e.doc._.modifiers.append(["Nomod"])
                if "MeasuredProperty" in annot:
                    propID = -1 
                    e.doc._.MP.append((annot["MeasuredProperty"]["span"].start,annot["MeasuredProperty"]["span"].end- 1,))
                    e.doc._.qa_mp_rel.append((len(e.doc._.QA)-1,len(e.doc._.MP)-1))
                    
                    for y in annot["MeasuredProperty"]["span"]:
                        propID = y.i
                        y._.relationDoc = (y.i, "HasQuantity", spanQuant)
                        break

                    if "MeasuredEntity" in annot:
                        e.doc._.ME.append((annot["MeasuredEntity"]["span"].start,annot["MeasuredEntity"]["span"].end- 1,))
                        e.doc._.mp_me_rel.append((len(e.doc._.MP)-1,len(e.doc._.ME)-1))
                        if propID != -1:
                            for y in annot["MeasuredEntity"]["span"]:
                                y._.relationDoc = (y.i, "HasPropety", propID)
                                break
                    else:
                        pass
                        #print("error no mesured entity, but property",e.name, annot)

                elif "MeasuredEntity" in annot:
                    e.doc._.ME.append((annot["MeasuredEntity"]["span"].start,annot["MeasuredEntity"]["span"].end- 1,))
                    e.doc._.qa_me_rel.append((len(e.doc._.QA)-1,len(e.doc._.ME)-1))
                    for y in annot["MeasuredEntity"]["span"]:
                        y._.relationDoc = (y.i, "HasQuantity", spanQuant)
                        break

                try: 
                    for y in annot["Qualifier"]["span"]:
                        y._.relationDoc = (y.i, "Qualifies", spanQuant)
                        e.doc._.QL.append((annot["Qualifier"]["span"].start,annot["Qualifier"]["span"].end- 1,))
                        e.doc._.qa_ql_rel.append((len(e.doc._.QA)-1,len(e.doc._.QL)-1))
                        break
                except KeyError:
                    pass
                

                #at the sentence level
                if len(annot["sentences"]) == 1:
                    sent = annot["sentences"][0]
                    ss = annot["sentences"][0].start
                    spanQuant = annot["Quantity"]["span"].start
                    sent._.QA.append((annot["Quantity"]["span"].start - ss,annot["Quantity"]["span"].end - ss - 1,))
                    try:
                        sent._.modifiers.append(annot["Quantity"]["other"]["mods"])
                    except KeyError:
                        sent._.modifiers.append(["Nomod"])

                    if "MeasuredProperty" in annot:
                        propID = -1
                        sent._.MP.append((annot["MeasuredProperty"]["span"].start - ss,annot["MeasuredProperty"]["span"].end - ss - 1,))
                        sent._.qa_mp_rel.append((len(sent._.QA)-1,len(sent._.MP)-1))
                        for y in annot["MeasuredProperty"]["span"]:
                            propID = y.i
                            y._.relationSent = (y.i - ss, "HasQuantity", spanQuant - ss)
                            break
                        if "MeasuredEntity" in annot:
                            sent._.ME.append((annot["MeasuredEntity"]["span"].start - ss,annot["MeasuredEntity"]["span"].end - ss - 1,))
                            sent._.mp_me_rel.append((len(sent._.MP)-1,len(sent._.ME)-1))
                            if propID != -1:
                                for y in annot["MeasuredEntity"]["span"]:
                                    y._.relationSent = (y.i - ss, "HasPropety", propID - ss)
                                    break
                        else:
                            pass
                            #print("error no mesured entity, but property",e.name, annot)
                    elif "MeasuredEntity" in annot:
                        sent._.ME.append((annot["MeasuredEntity"]["span"].start - ss,annot["MeasuredEntity"]["span"].end - ss - 1,))
                        sent._.qa_me_rel.append((len(sent._.QA)-1,len(sent._.ME)-1))
                        for y in annot["MeasuredEntity"]["span"]:
                            y._.relationSent = (y.i - ss, "HasQuantity", spanQuant - ss)
                            break

                    try: 
                        for y in annot["Qualifier"]["span"]:
                            sent._.QL.append((annot["Qualifier"]["span"].start - ss,annot["Qualifier"]["span"].end - ss - 1,))
                            sent._.qa_ql_rel.append((len(sent._.QA)-1,len(sent._.QL)-1))
                            y._.relationSent = (y.i - ss, "Qualifies", spanQuant - ss)
                            break
                    except KeyError:
                        pass
                else:
                    doc = e.doc
                    sent = doc[annot["Quantity"]["span"].start].sent
                    ss = sent.start
                    sent._.QA.append((annot["Quantity"]["span"].start - ss,annot["Quantity"]["span"].end - ss - 1,))
                    try:
                        sent._.modifiers.append(annot["Quantity"]["other"]["mods"])
                    except KeyError:
                        sent._.modifiers.append(["Nomod"])
                    if "MeasuredProperty" in annot:
                        sent = doc[annot["MeasuredProperty"]["span"].start].sent
                        ss = sent.start
                        sent._.MP.append((annot["MeasuredProperty"]["span"].start - ss,annot["MeasuredProperty"]["span"].end - ss - 1,))

                        if "MeasuredEntity" in annot:
                            sent = doc[annot["MeasuredEntity"]["span"].start].sent
                            ss = sent.start
                            sent._.ME.append((annot["MeasuredEntity"]["span"].start - ss,annot["MeasuredEntity"]["span"].end - ss - 1,))
                        else:
                            pass
                            #print("error no mesured entity, but property",e.name, annot)

                    elif "MeasuredEntity" in annot:
                        sent = doc[annot["MeasuredEntity"]["span"].start].sent
                        ss = sent.start
                        sent._.ME.append((annot["MeasuredEntity"]["span"].start - ss,annot["MeasuredEntity"]["span"].end - ss - 1,))

                    if "Qualifier" in annot:
                        sent = doc[annot["Qualifier"]["span"].start].sent
                        ss = sent.start
                        sent._.QL.append((annot["Qualifier"]["span"].start - ss,annot["Qualifier"]["span"].end - ss - 1,))




    def getData(self):
        if not os.path.isdir("generatedData"):
            os.mkdir("generatedData")
        else:
            os.system("rm generatedData/*")

        allP = open(os.path.join("generatedData", "allP.tsv"),"w",encoding="utf-8")
        quantP = open(os.path.join("generatedData", "quantP.tsv"),"w",encoding="utf-8")
        qualP = open(os.path.join("generatedData", "qualP.tsv"),"w",encoding="utf-8")
        meP = open(os.path.join("generatedData", "meP.tsv"),"w",encoding="utf-8")
        mpP = open(os.path.join("generatedData", "mpP.tsv"),"w",encoding="utf-8")

        for x in self.data.values():
            for token in x.doc:
                allP.write(token.text+"\t"+token._.all+"\t"+token.tag_+"\n")
                quantP.write(token.text+"\t"+token._.quant+"\t"+token.tag_+"\n")
                qualP.write(token.text+"\t"+token._.qual+"\t"+token.tag_+"\n")
                meP.write(token.text+"\t"+token._.me+"\t"+token.tag_+"\n")
                mpP.write(token.text+"\t"+token._.mp+"\t"+token.tag_+"\n")
            allP.write("\n")
            quantP.write("\n")
            qualP.write("\n")
            meP.write("\n")
            mpP.write("\n")

        allP.close()
        quantP.close()
        qualP.close()
        meP.close()
        mpP.close()

        allS = open(os.path.join("generatedData", "allS.tsv"),"w",encoding="utf-8")
        quantS = open(os.path.join("generatedData", "quantS.tsv"),"w",encoding="utf-8")
        qualS = open(os.path.join("generatedData", "qualS.tsv"),"w",encoding="utf-8")
        meS = open(os.path.join("generatedData", "meS.tsv"),"w",encoding="utf-8")
        mpS = open(os.path.join("generatedData", "mpS.tsv"),"w",encoding="utf-8")

        for x in self.data.values():

            for sent in x.doc.sents:
                for token in sent:
                    allS.write(token.text+"\t"+token._.all+"\t"+token.tag_+"\n")
                    quantS.write(token.text+"\t"+token._.quant+"\t"+token.tag_+"\n")
                    qualS.write(token.text+"\t"+token._.qual+"\t"+token.tag_+"\n")
                    meS.write(token.text+"\t"+token._.me+"\t"+token.tag_+"\n")
                    mpS.write(token.text+"\t"+token._.mp+"\t"+token.tag_+"\n")

                allS.write("\n")
                quantS.write("\n")
                qualS.write("\n")
                meS.write("\n")
                mpS.write("\n")

        allS.close()
        quantS.close()
        qualS.close()
        meS.close()
        mpS.close()


    def readDataEval(self, filepath):

        filenames = os.listdir(os.path.join(filepath,"eval","text"))
        filenames = [x for x in filenames if x[-4:] == ".txt"]

        assert(len(filenames) == 135)

        data = {}

        for fn in tqdm(filenames):
            data[fn[:-4]] = Exerpt(
                    name = fn[:-4],
                    txt = str(open(os.path.join(filepath,"eval","text", fn), "r", encoding = "utf-8").read()),
                    ann = None,
                    tsv = None,
                    evaluation = True
                )

        return data


        
        
    


    def readData(self, filepath):

        readFileRaw = lambda path : str(open(path, "r", encoding = "utf-8").read())
        TRAINPATH = os.path.join(filepath,"train")
        TRIALPATH = os.path.join(filepath,"trial")

        data = {}
        
        t1 = time.time()
        logging.info("Processing Trial Files")
        #load all trial data
        for fn in tqdm(os.listdir(os.path.join(TRIALPATH,"txt"))):
            if fn.endswith('.txt'):
                data[fn[:-4]] = Exerpt(
                    fn[:-4],
                    readFileRaw(os.path.join(TRIALPATH, "txt", fn[:-4] + ".txt")),
                    readFileRaw(os.path.join(TRIALPATH, "ann", fn[:-4] + ".ann")),
                    pd.read_csv(os.path.join(TRIALPATH, "tsv", fn[:-4] + ".tsv"), "\t", header = 0 )
                )

        logging.info("Processing Train Files")
        #load all train data
        for fn in tqdm([x for x in os.listdir(os.path.join(TRAINPATH,"tsv"))]):
            if fn.endswith('.tsv'):
                data[fn[:-4]] = Exerpt(
                    fn[:-4],
                    readFileRaw(os.path.join(TRAINPATH, "text", fn[:-4] + ".txt")),
                    "none",
                    pd.read_csv(os.path.join(TRAINPATH, "tsv", fn[:-4] + ".tsv"), "\t", header = 0 )
                )
            
        t2 = time.time()
        logging.info("{} Seconds elapsed".format(t2-t1))
        logging.info("{} Minutes elapsed".format((t2-t1)/60))

        return data


    def getAscii(self, filepath):
        if os.path.isdir(filepath):
            pass
        else:
            os.mkdir(filepath)

        for x in self.data.values():
            x.getAscii(filepath)

    def getAsciiConstituent(self, filepath, constituency = False):
        if os.path.isdir(filepath):
            pass
        else:
            os.mkdir(filepath)
            
        for x in self.data.values():
            x.getAsciiConstituent(filepath, constituency)

    def getGateJson(self, filepath):
        if os.path.isdir(filepath):
            pass
        else:
            os.mkdir(filepath)
            
        for x in self.data.values():
            x.getGateJson(filepath)

    

    def getLatexEval(self):
        temp = self.evaluateDocs()
        accum = "Type&Precision&Recall&F1\\\\\n\\hline\n"
        for x in ["Quantity","ME"]:
            accum += "{}&{}&{}&{}\\\\\n\\hline\n".format(x,temp[x+"Precision"],temp[x+"Recall"],temp[x+"F1"])
        return accum

    def getConfusionMatrix(self,tpe):
        temp = self.evaluateDocs()
        accum = "&Condition Positive&Condition Negative\\\\\n\\hline\n"
        accum +="Predicted {}&{}&{}\\\\\n\\hline\n".format(tpe,temp["h0Count"][tpe],temp["h0Count"]["total"]-temp["h0Count"][tpe])
        if tpe in ["Unit","Number"]:
            accum +="Predicted Not {}&{}&{}\\\\\n\\hline\n".format(tpe,temp["goldCount"]["Quantity"]-temp["h0Count"][tpe],0)
        else:
            accum +="Predicted Not {}&{}&{}\\\\\n\\hline\n".format(tpe,temp["goldCount"][tpe]-temp["h0Count"][tpe],0)
        return accum



    def getAllOfTSV(self, annotType,category):
        if annotType not in ["Quantity","MeasuredEntity","MeasuredProperty","Qualifier"]:
            logging.error("Error Occured in exerptController class getAllOfTSV()")
            return []

        tempList = []
        for e in self.data.values():
            for x in e.measurements.values():
                try:
                    tempList.append(x[annotType][category])
                except KeyError:
                    pass 

        return tempList 

    def extractQuantOther(self):
        quanto = self.getAllOfTSV("Quantity","other")

        def getDict(s):
            s.replace("\"","")
            s.replace("'","")
            temp = word_tokenize(s)
            g = ""
            x=0
            while(x<len(temp)):
                if temp[x].isalnum() and (temp[x+1].isalnum() or temp[x+1] in [".","/"]):
                    temptemp= ""
                    while(temp[x] not in ["}",","]):
                        temptemp+=temp[x]
                        x+=1
                    g+="\""+temptemp+"\""
                    x-=1
                elif temp[x].isalpha() :
                    g+="\""+temp[x]+"\""
                elif (temp[x-1] == ":" and temp[x+1] == ",") or (temp[x-1] == ":" and temp[x+1] == "}"):
                    g+="\""+temp[x]+"\""
                else:
                    g+=temp[x]
                x+=1
            return g

        dics = []
        for x in range(len(quanto)):
            if(type(quanto[x]) == str):
                try:
                    dics.append(json.loads(quanto[x]))
                except Exception:
                    try:
                        temp = word_tokenize(quanto[x])
                        dics.append(json.loads(getDict(quanto[x])))
                    except Exception:
                        print(quanto[x],"Not captured by extractQuantOther()")
                        #DEBUG
                        pass
        
        allmods = []
        allunits = []
        for x in dics:
            try:
                for y in x["mods"]:
                    allmods.append(y)
            except Exception:
                pass
        
            try:
                allunits.append(x["unit"])
            except Exception:
                pass

        
        return list({k:1 for k in allmods}.keys()), list({k:1 for k in allunits}.keys())


    def getFolds(self, fold, div = 8):
        """
        Splits the given data into a different fold based on input
        """
        if fold < 1  or fold > div:
            print("Incorrect div to fold number encountered")
            return None, None
        data = list(self.data.keys())
        testSize = math.floor(len(data)/div)
        beforeTest = data[:(fold-1)*testSize]
        test = data[(fold-1)*testSize:fold*testSize]
        afterTest = data[fold*testSize:]
        return test, beforeTest + afterTest


    def getEncodingSent(self, fold, syspath, skip=[], div = 8, test=None,train=None):
        if test == None or train == None:
            test, train = self.getFolds(fold, div)

        datapath = os.path.join(syspath,"data-fold{}".format(fold))
        if not os.path.isdir(datapath):
            os.mkdir(datapath)

        print(os.path.join(syspath,"data-fold{}".format(fold), "train.txt"))

        with open(os.path.join(syspath,"data-fold{}".format(fold), "train.txt"), "w", encoding="utf-8") as f:
            for x in train:
                f.write(x+"\n")
            f.close()

        with open(os.path.join(syspath,"data-fold{}".format(fold), "test.txt"), "w", encoding="utf-8") as f:
            for x in test:
                f.write(x+"\n")
            f.close()
        

        train_allS = open(os.path.join(datapath, "train_allS.tsv"),"w",encoding="utf-8")
        train_quantS = open(os.path.join(datapath, "train_quantS.tsv"),"w",encoding="utf-8")
        train_qualS = open(os.path.join(datapath, "train_qualS.tsv"),"w",encoding="utf-8")
        train_meS = open(os.path.join(datapath, "train_meS.tsv"),"w",encoding="utf-8")
        train_mpS = open(os.path.join(datapath, "train_mpS.tsv"),"w",encoding="utf-8")

        
        for i,y in enumerate(train):
            if y in skip: 
                continue
            count=0
            sOffset=0
            for sent in self.data[y].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    train_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                
                train_allS.write("@@annotations@@\n")
                temp = ["QA"]
                for i,x in enumerate(sent._.QA):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["ME"]
                for i,x in enumerate(sent._.ME):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["MP"]
                for i,x in enumerate(sent._.MP):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["QL"]
                for i,x in enumerate(sent._.QL):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["QA_ME_Rel"]
                for i,x in enumerate(sent._.qa_me_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["QA_MP_Rel"]
                for i,x in enumerate(sent._.qa_mp_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["MP_ME_Rel"]
                for i,x in enumerate(sent._.mp_me_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["QA_QL_Rel"]
                for i,x in enumerate(sent._.qa_ql_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                train_allS.write("\t".join(temp)+"\n")
                temp = ["modifiers"]
                for i,x in enumerate(sent._.modifiers):
                    for mod in x:
                        temp.append("{}:{}".format(i,mod))
                train_allS.write("\t".join(temp)+"\n")
                train_allS.write("\t".join(["Doc",str(y)])+"\n")

                train_allS.write("\n")
                train_quantS.write("\n")
                train_qualS.write("\n")
                train_meS.write("\n")
                train_mpS.write("\n")

        train_allS.close()
        train_quantS.close()
        train_qualS.close()
        train_meS.close()
        train_mpS.close()

        test_allS = open(os.path.join(datapath, "test_allS.tsv"),"w",encoding="utf-8")
        test_quantS = open(os.path.join(datapath, "test_quantS.tsv"),"w",encoding="utf-8")
        test_qualS = open(os.path.join(datapath, "test_qualS.tsv"),"w",encoding="utf-8")
        test_meS = open(os.path.join(datapath, "test_meS.tsv"),"w",encoding="utf-8")
        test_mpS = open(os.path.join(datapath, "test_mpS.tsv"),"w",encoding="utf-8")

        

        for i,y in enumerate(test):
            if y in skip: 
                continue
            count=0
            sOffset=0
            for sent in self.data[y].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    test_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                
                test_allS.write("@@annotations@@\n")
                temp = ["QA"]
                for i,x in enumerate(sent._.QA):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["ME"]
                for i,x in enumerate(sent._.ME):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["MP"]
                for i,x in enumerate(sent._.MP):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["QL"]
                for i,x in enumerate(sent._.QL):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["QA_ME_Rel"]
                for i,x in enumerate(sent._.qa_me_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["QA_MP_Rel"]
                for i,x in enumerate(sent._.qa_mp_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["MP_ME_Rel"]
                for i,x in enumerate(sent._.mp_me_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["QA_QL_Rel"]
                for i,x in enumerate(sent._.qa_ql_rel):
                    temp.append("{}:({},{})".format(i,x[0],x[1]))
                test_allS.write("\t".join(temp)+"\n")
                temp = ["modifiers"]
                for i,x in enumerate(sent._.modifiers):
                    for mod in x:
                        temp.append("{}:{}".format(i,mod))
                test_allS.write("\t".join(temp)+"\n")
                test_allS.write("\t".join(["Doc",str(y)])+"\n")
                
                test_allS.write("\n")
                test_quantS.write("\n")
                test_qualS.write("\n")
                test_meS.write("\n")
                test_mpS.write("\n")

        test_allS.close()
        test_quantS.close()
        test_qualS.close()
        test_meS.close()
        test_mpS.close()

    def getEncodingDoc(self, fold, syspath, skip=[], div = 8):
        test, train = self.getFolds(fold, div)

        with open(os.path.join(syspath,"data-fold{}".format(fold),"train.txt"), "w", encoding="utf-8") as f:
            for x in train:
                f.write(x+"\n")

        with open(os.path.join(syspath,"data-fold{}".format(fold),"test.txt"), "w", encoding="utf-8") as f:
            for x in test:
                f.write(x+"\n")

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")

        train_allD = open(os.path.join(datapath, "train_allD.tsv"),"w",encoding="utf-8")
        train_quantD = open(os.path.join(datapath, "train_quantD.tsv"),"w",encoding="utf-8")
        train_qualD = open(os.path.join(datapath, "train_qualD.tsv"),"w",encoding="utf-8")
        train_meD = open(os.path.join(datapath, "train_meD.tsv"),"w",encoding="utf-8")
        train_mpD = open(os.path.join(datapath, "train_mpD.tsv"),"w",encoding="utf-8")
        
        for i,x in enumerate(train):
            if x in skip: 
                continue
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    train_allD.write("\t".join([token.text,token._.all,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_quantD.write("\t".join([token.text,token._.quant,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_qualD.write("\t".join([token.text,token._.qual,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_meD.write("\t".join([token.text,token._.me,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_mpD.write("\t".join([token.text,token._.mp,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")

            train_allD.write("@@annotations@@\n")
            temp = ["QA"]
            for i,y in enumerate(self.data[x].doc._.QA):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["ME"]
            for i,y in enumerate(self.data[x].doc._.ME):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["MP"]
            for i,y in enumerate(self.data[x].doc._.MP):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["QL"]
            for i,y in enumerate(self.data[x].doc._.QL):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["QA_ME_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_me_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["QA_MP_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_mp_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["MP_ME_Rel"]
            for i,y in enumerate(self.data[x].doc._.mp_me_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            temp = ["QA_QL_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_ql_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            train_allD.write("\t".join(temp)+"\n")
            train_allD.write("\t".join(["Doc",str(x)])+"\n")
                

            train_allD.write("\n")
            train_quantD.write("\n")
            train_qualD.write("\n")
            train_meD.write("\n")
            train_mpD.write("\n")

        train_allD.close()
        train_quantD.close()
        train_qualD.close()
        train_meD.close()
        train_mpD.close()

        test_allD = open(os.path.join(datapath, "test_allD.tsv"),"w",encoding="utf-8")
        test_quantD = open(os.path.join(datapath, "test_quantD.tsv"),"w",encoding="utf-8")
        test_qualD = open(os.path.join(datapath, "test_qualD.tsv"),"w",encoding="utf-8")
        test_meD = open(os.path.join(datapath, "test_meD.tsv"),"w",encoding="utf-8")
        test_mpD = open(os.path.join(datapath, "test_mpD.tsv"),"w",encoding="utf-8")


        for i,x in enumerate(test):
            if x in skip: 
                continue
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    test_allD.write("\t".join([token.text,token._.all,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_quantD.write("\t".join([token.text,token._.quant,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_qualD.write("\t".join([token.text,token._.qual,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_meD.write("\t".join([token.text,token._.me,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_mpD.write("\t".join([token.text,token._.mp,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")

            test_allD.write("@@annotations@@\n")
            temp = ["QA"]
            for i,y in enumerate(self.data[x].doc._.QA):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["ME"]
            for i,y in enumerate(self.data[x].doc._.ME):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["MP"]
            for i,y in enumerate(self.data[x].doc._.MP):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["QL"]
            for i,y in enumerate(self.data[x].doc._.QL):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["QA_ME_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_me_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["QA_MP_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_mp_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["MP_ME_Rel"]
            for i,y in enumerate(self.data[x].doc._.mp_me_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            temp = ["QA_QL_Rel"]
            for i,y in enumerate(self.data[x].doc._.qa_ql_rel):
                temp.append("{}:({},{})".format(i,y[0],y[1]))
            test_allD.write("\t".join(temp)+"\n")
            test_allD.write("\t".join(["Doc",str(x)])+"\n")

            
            test_allD.write("\n")
            test_quantD.write("\n")
            test_qualD.write("\n")
            test_meD.write("\n")
            test_mpD.write("\n")

        test_allD.close()
        test_quantD.close()
        test_qualD.close()
        test_meD.close()
        test_mpD.close()


    def writeDataMods(self,fold,syspath,div=8,window=0,test=None, train=None):
        def getContext(span,window):
            return span.doc[max(0,span.start-window):min(span.end+window,len(span.doc))]

        if test == None or train == None:
            test, train = self.getFolds(fold, div)

        #f = open("errors.txt","w",encoding="utf-8")
        train_dat = [] 
        for x in train:
            for y in self.data[x].doc._.meAnnots.values():
                try:
                    train_dat.append({"span":[z.text for z in getContext(y["Quantity"]["span"],window)],"type":y["Quantity"]["other"]["mods"]})
                except KeyError: 
                    try:
                        train_dat.append({"span":[z.text for z in getContext(y["Quantity"]["span"],window)],"type":["NOMOD"]})
                    except KeyError: 
                        pass


        test_dat = [] 
        for x in test:
            for y in self.data[x].doc._.meAnnots.values():
                try:
                    test_dat.append({"span":[z.text for z in getContext(y["Quantity"]["span"],window)],"type":y["Quantity"]["other"]["mods"]})
                except KeyError: 
                    try:
                        test_dat.append({"span":[z.text for z in getContext(y["Quantity"]["span"],window)],"type":["NOMOD"]})
                    except KeyError: 
                        pass

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)

        json.dump(test_dat,open(os.path.join(datapath,"test-mod-data-{}.json".format(window)),"w",encoding="utf-8"),indent=4)
        json.dump(train_dat,open(os.path.join(datapath,"train-mod-data-{}.json".format(window)),"w",encoding="utf-8"),indent=4)


    def writeDataPaths(self,fold,syspath,div=8):
        def getDataForDoc(doc):
            sent_data = []
            for sent in doc.sents:
                cds = []
                for token in sent:
                    if token.tag_ == "CD":
                        cds.append(token)
                    
                a = graph.Graph(sent)
                tempDict = {"sentence":[x.text for x in sent]}
                for qa in cds:
                    key = qa.i
                    for tok in sent:
                        path = a.getShortestPath(source=a.getNode(qa),target=a.getNode(tok))
                        if path == []:
                            path = ['self']
                        tempVal = {
                            "path":path,
                            "target":tok._.all if qa._.annotId == tok._.annotId else "o"
                        }
                        
                        try:
                            tempDict[key].append(tempVal)
                        except KeyError:
                            tempDict[key] = [tempVal]
                
                
                sent_data.append(tempDict)
            return sent_data


        test, train = self.getFolds(fold, div)

        #f = open("errors.txt","w",encoding="utf-8")
        train_dat = [] 
        for x in train:
            train_dat = train_dat + getDataForDoc(self.data[x].doc)


        test_dat = [] 
        for x in test:
            test_dat = test_dat + getDataForDoc(self.data[x].doc)

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")


        json.dump(test_dat,open(os.path.join(datapath,"test-path-data.json"),"w",encoding="utf-8"),indent=4)
        json.dump(train_dat,open(os.path.join(datapath,"train-path-data.json"),"w",encoding="utf-8"),indent=4)



    def writeDataPaths2(self,fold,syspath,div=8):
        def getDataForDoc(doc):
            sent_data = []
            for sent in doc.sents:
                cds = []
                for token in sent:
                    if token._.all == "QA":
                        cds.append(token)
                        #print("a",token,token._.all)
                    
                a = graph.Graph(sent)
                tempDict = {"sentence":[x.text for x in sent]}
                for qa in cds:
                    key = qa.i
                    for tok in sent:
                        tempVal = {
                            "path":a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok)),
                            "target":tok._.all if qa._.annotId == tok._.annotId else "o"
                        }
                        
                        try:
                            tempDict[key].append(tempVal)
                        except KeyError:
                            tempDict[key] = [tempVal]
                
                
                sent_data.append(tempDict)
            return sent_data


        test, train = self.getFolds(fold, div)

        #f = open("errors.txt","w",encoding="utf-8")
        train_dat = [] 
        for x in train:
            train_dat = train_dat + getDataForDoc(self.data[x].doc)


        test_dat = [] 
        for x in test:
            test_dat = test_dat + getDataForDoc(self.data[x].doc)

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")


        json.dump(test_dat,open(os.path.join(datapath,"test-path-data-2.json"),"w",encoding="utf-8"),indent=4)
        json.dump(train_dat,open(os.path.join(datapath,"train-path-data-2.json"),"w",encoding="utf-8"),indent=4)        
    


    def writeDataPathsNoun(self,fold,syspath,div=8):
            def getDataForDoc(doc):
                sent_data = []
                for sent in doc.sents:
                    cds = []
                    for token in sent:
                        if token._.all == "QA":
                            cds.append(token)
                            #print("a",token,token._.all)
                        
                    a = graph.Graph(sent)
                    tempDict = {"sentence":[x.text for x in sent]}
                    for qa in cds:
                        key = qa.i
                        
                        for tok in sent:
                            if tok._.all != "QA":
                                tempVal = {
                                    "path":a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok)),
                                    "target":tok._.all if qa._.annotId == tok._.annotId else "o"
                                }
                                
                                try:
                                    tempDict[key].append(tempVal)
                                except KeyError:
                                    tempDict[key] = [tempVal]
                    
                    
                    sent_data.append(tempDict)
                return sent_data


            test, train = self.getFolds(fold, div)

            #f = open("errors.txt","w",encoding="utf-8")
            train_dat = [] 
            for x in train:
                train_dat = train_dat + getDataForDoc(self.data[x].doc)


            test_dat = [] 
            for x in test:
                test_dat = test_dat + getDataForDoc(self.data[x].doc)

            datapath = os.path.join(syspath,"data-fold{}".format(fold))

            if not os.path.isdir(datapath):
                os.mkdir(datapath)


            json.dump(test_dat,open(os.path.join(datapath,"test-path-data-NOUN.json"),"w",encoding="utf-8"),indent=4)
            json.dump(train_dat,open(os.path.join(datapath,"train-path-data-NOUN.json"),"w",encoding="utf-8"),indent=4)   

    def writeDataPathsSmall(self,fold,syspath,div=8,train=None,test=None):

        

        def getDataForDoc(doc):
            def getContext(span,window):
                sent = span.sent
                return span.doc[max(sent.start,span.i-window-1):min(span.i+window,sent.end- 1)]

            def getLastNoun(span):
                span = [tok for tok in span]
                if span[-1].tag_ in ["NNS","NN","NNP"]:
                    return span[-1]
                print([(x,x.tag_) for x in span])
                return span[-1]

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

            def getHeadNouns(quant):
                headNouns= [] 
                for token in quant.sent:
                    if intersectSpanSpan(token,quant):
                        continue
                    try: 
                        if token.tag_ in ["NNS","NN","NNP"] and token._.all == "o":
                            if intersectSpanSpan(doc[token.i+1],quant):
                                headNouns.append(token)
                            elif doc[token.i+1].tag_ in ["NNS","NN","NNP"]:
                                continue
                            else:
                                headNouns.append(token)

                    except IndexError:
                        continue

                return headNouns

                    
            sent_data = []
            for x in doc._.meAnnots.values():
                if len(x["sentences"]) > 1:
                    continue

                a = graph.Graph(x["sentences"][0])
                qa = getSource(x["Quantity"]["span"])

                for key in x.keys():
                    if key in ["Quantity", "sentences"]:
                        continue
                    if [y for y in x[key]["span"]] == []:
                        continue
                    tok = getLastNoun(x[key]["span"])
                    if tok._.all in ["QL","ME","MP"]:
                        if len(a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok))) <= 5:
                            tempVal = {
                                        "path":a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok)),
                                        "quantSpan":[x.text for x in x["Quantity"]["span"]],
                                        "nounSpan":[x.text for x in getContext(tok,2)],
                                        "target":tok._.all 
                                    }
                            if tempVal["target"] == "o":
                                print(x)
                                print(tok)
                                print(qa._.annotId)
                                print(tok._.annotId)
                                print(qa[0]._.annotId)
                            sent_data.append(tempVal)

                headNouns = getHeadNouns(qa)
                print(headNouns)
                if headNouns == []:
                    continue
                else:
                    random.shuffle(headNouns)
                    tok = headNouns[0]
                    if len(a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok))) <= 5:
                        tempVal = {
                                    "path":a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok)),
                                    "quantSpan":[x.text for x in x["Quantity"]["span"]],
                                    "nounSpan":[x.text for x in getContext(tok,2)],
                                    "target":tok._.all 
                                }
                        sent_data.append(tempVal)
                    if len(headNouns) > 1:
                        tok = headNouns[1]
                        if len(a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok))) <= 5:
                            tempVal = {
                                        "path":a.getShortestPathToken(source=a.getNode(qa),target=a.getNode(tok)),
                                        "quantSpan":[x.text for x in x["Quantity"]["span"]],
                                        "nounSpan":[x.text for x in getContext(tok,2)],
                                        "target":tok._.all 
                                    }
                            sent_data.append(tempVal)


            return sent_data


        if train == None or test == None:
            test, train = self.getFolds(fold, div)

        #f = open("errors.txt","w",encoding="utf-8")
        train_dat = [] 
        for x in train:
            train_dat = train_dat + getDataForDoc(self.data[x].doc)


        test_dat = [] 
        for x in test:
            test_dat = test_dat + getDataForDoc(self.data[x].doc)

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)


        json.dump(test_dat,open(os.path.join(datapath,"test-path-data-Small.json"),"w",encoding="utf-8"),indent=4)
        json.dump(train_dat,open(os.path.join(datapath,"train-path-data-Small.json"),"w",encoding="utf-8"),indent=4)   



    def writeDataPruner(self,fold,syspath,div=8):
            def getDataForDoc(doc):
                data = []
                for x in doc._.meAnnots.values():
                    try:
                        data.append({"span":[tok.text for tok in x["MeasuredEntity"]["span"]],"type":["ME"]})
                    except:
                        pass
                    
                    try:
                        data.append({"span":[tok.text for tok in x["MeasuredProperty"]["span"]],"type":["MP"]})
                    except:
                        pass
                    
                    try:
                        data.append({"span":[tok.text for tok in x["Quantity"]["span"]],"type":["QA"]})
                    except:
                        pass
                    
                    try:
                        data.append({"span":[tok.text for tok in x["Qualifier"]["span"]],"type":["QL"]})
                    except:
                        pass
                count = 0

                while(count < 5):
                    count+=1
                    b = np.random.randint(low=0,high=len(doc))
                    if doc[b]._.all == "o":
                        data.append({"span":[doc[b].text],"type":["o"]})
                    
                    
                return data


            test, train = self.getFolds(fold, div)

            #f = open("errors.txt","w",encoding="utf-8")
            train_dat = [] 
            for x in train:
                train_dat += getDataForDoc(self.data[x].doc)

            test_dat = [] 
            for x in test:
                test_dat += getDataForDoc(self.data[x].doc)

            datapath = os.path.join(syspath,"data-fold{}".format(fold))

            if not os.path.isdir(datapath):
                os.mkdir(datapath)


            json.dump(test_dat,open(os.path.join(datapath,"test-span.json"),"w",encoding="utf-8"),indent=4)
            json.dump(train_dat,open(os.path.join(datapath,"train-span.json"),"w",encoding="utf-8"),indent=4)   

    

    def writeDataMatcher(self,fold,syspath,div=8,context=0):
        
        def getDataForDoc(doc,context):
            def withinSentence(span1,span2):
                doc = span1.doc
                s1 = span1.sent
                s2 = span2.sent
                i1,i2 = -10,500
                s = [x for x in doc.sents]
                print(len(s))
                for i, sent in enumerate(s):
                    if intersectSpanSpan(s1,sent):
                        i1=i

                    if intersectSpanSpan(s2,sent):
                        i2=i
                        
                return abs(i1-i2) <= 1

            def getContext(span,window):
                return span.doc[max(0,span.start-window):min(span.end+window,len(span.doc))]

            c = context
            data=[]
            temp = doc._.meAnnots
            for key in temp.keys():
                for kk in temp[key].keys():
                    if kk not in ["Quantity","sentences"]:
                        data.append(([x.text for x in getContext(temp[key]["Quantity"]["span"],c)],
                                    [y.text for y in getContext(temp[key][kk]["span"],c)],
                                    "match"))
                        
                for key2 in temp.keys():
                    if key2 != key:
                        for kk2 in temp[key2].keys():
                            if kk2 not in ["Quantity","sentences"]:
                                if(withinSentence(temp[key]["Quantity"]["span"],temp[key2][kk2]["span"])):
                                    data.append(([x.text for x in getContext(temp[key]["Quantity"]["span"],c)],
                                                [y.text for y in getContext(temp[key2][kk2]["span"],c)],
                                                "nomatch"))
            return data


        test, train = self.getFolds(fold, div)

        #f = open("errors.txt","w",encoding="utf-8")
        train_dat = [] 
        for x in train:
            train_dat += getDataForDoc(self.data[x].doc,context)

        test_dat = [] 
        for x in test:
            test_dat += getDataForDoc(self.data[x].doc,context)

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)


        json.dump(test_dat,open(os.path.join(datapath,"test-match-{}.json".format(context)),"w",encoding="utf-8"),indent=4)
        json.dump(train_dat,open(os.path.join(datapath,"train-match-{}.json".format(context)),"w",encoding="utf-8"),indent=4)   



