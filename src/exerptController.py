import logging
import time
from tqdm import tqdm
import pandas as pd
import os
from src.exerpt import Exerpt
import json
import spacy
from src.lib.helpers import createAnnotation, createFeature, createNode
import math
from nltk.tokenize import word_tokenize
import random
from spacy.tokens import Token


class ExerptController:
    """
    Description: Class containing all exerpts that encapsulates all the logic needed to deal with them 
    """


    def __init__(self, filepath):
        if(os.path.isdir(filepath)):
            self.data = self.readData(filepath)
        else:
            self.data = {}

        self.annotateData()

        self.getData()

    def annotateData(self):
        Token.set_extension("all", default="o", force=True)
        Token.set_extension("quant", default="o", force=True)
        Token.set_extension("qual", default="o", force=True)
        Token.set_extension("me", default="o", force=True)
        Token.set_extension("mp", default="o", force=True)
        tempDict = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
        for x in self.data.values():
            for annot in x.doc._.meAnnots.values():
                for key in annot.keys():
                    if key != "sentences":
                        tempSpanlen = len(annot[key])
                        count = -1
                        prefix = ""
                        for token in annot[key]:
                            if count == 0:
                                prefix = "B_"
                            elif count > 0 and count < tempSpanlen-1:
                                prefix = "I_"
                            elif count == -1:
                                prefix = ""
                            else:
                                prefix = "E_"

                            

                            token._.all = prefix+tempDict[key]
                            if key == "MeasuredEntity":
                                token._.me = prefix+tempDict[key]
                            elif key == "Quantity":
                                token._.quant = prefix+tempDict[key]
                            elif key == "Qualifier":
                                token._.qual = prefix+tempDict[key]
                            elif key == "MeasuredProperty":
                                token._.mp = prefix+tempDict[key]
                            #count+=1

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
                        pd.read_csv(os.path.join(TRIALPATH, "tsv", fn[:-4] + ".tsv"), "\t", header = 0 ),
                        json.load(open(os.path.join(TRIALPATH, "grobid", fn[:-4] + ".grobid")))
                    )

        logging.info("Processing Train Files")
        #load all train data
        for fn in tqdm([x for x in os.listdir(os.path.join(TRAINPATH,"text")) if x[:-4]+".tsv" in os.listdir(os.path.join(TRAINPATH,"tsv"))]):
                if fn.endswith('.txt'):
                    data[fn[:-4]] = Exerpt(
                        fn[:-4],
                        readFileRaw(os.path.join(TRAINPATH, "text", fn[:-4] + ".txt")),
                        "none",
                        pd.read_csv(os.path.join(TRAINPATH, "tsv", fn[:-4] + ".tsv"), "\t", header = 0 ),
                        json.load(open(os.path.join(TRAINPATH, "grobid", fn[:-4] + ".grobid")))
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

    def evaluateDocs(self):
        goldCount = {
            "Quantity": 0,
            "MeasuredEntity" : 0,
            "MeasuredProperty" : 0,
            "Qualifier" : 0       
        }

        h0Count = {
            "Number":0,
            "Unit":0,
            "MeasuredEntity":0,
            "total":0
        }

        counts={
            "goldCount" : goldCount,
            "h0Count" : h0Count
        }

        for e in self.data.values():
            for index, row in e.tsv.iterrows():
                    counts["goldCount"][row["annotType"]] += 1
            counts["h0Count"]["Number"] += len(e.doc._.h0NumberTps)
            counts["h0Count"]["Unit"] += len(e.doc._.h0UnitTps)
            counts["h0Count"]["MeasuredEntity"] += len(e.doc._.h0MeasuredEntityTps)
            counts["h0Count"]["total"] += len(e.doc._.h0Measurements)
            
            
        counts["QuantityPrecision"] = counts["h0Count"]["Number"]/counts["h0Count"]["total"]
        counts["QuantityRecall"] = counts["h0Count"]["Number"]/counts["goldCount"]["Quantity"]
        counts["QuantityF1"] = 2*(counts["QuantityRecall"]*counts["QuantityPrecision"])/(counts["QuantityRecall"]+counts["QuantityPrecision"])
        counts["MEPrecision"] = counts["h0Count"]["MeasuredEntity"]/counts["h0Count"]["total"]
        counts["MERecall"] = counts["h0Count"]["MeasuredEntity"]/counts["goldCount"]["Quantity"]
        counts["MEF1"] = 2*(counts["MERecall"]*counts["MEPrecision"])/(counts["MERecall"]+counts["MEPrecision"])

        return counts

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



     


    def getXML(self, docname, filepath=None, numberOfDocs=300):
        if filepath == None:
            pass
        elif os.path.isdir(filepath):
            pass
        else:
            os.mkdir(filepath)

        pathToText = ""
        if filepath == None:
            pathToText = f"{docname}TEXT.txt"
            xmlFile = open(f"{docname}.xml", "w", encoding = "utf-8")
            txtFile = open(pathToText, "w", encoding = "utf-8")
            
        else:
            pathToText = os.path.join(filepath,f"{docname}TEXT.xml")
            xmlFile = open(os.path.join(filepath,f"{docname}.xml"), "w", encoding = "utf-8")
            txtFile = open(pathToText, "w", encoding = "utf-8")
        
        

        xmlFile.write("""<?xml version='1.0' encoding='utf-8'?>
        <GateDocument version="3">
        <GateDocumentFeatures>""")
        createFeature("gate.SourceURL",os.path.join(os.getcwd(), pathToText),xmlFile)
        createFeature("MimeType","text/plain",xmlFile)
        createFeature("docNewLineType","",xmlFile)
        xmlFile.write("\n</GateDocumentFeatures>\n\n")
        xmlFile.write("<TextWithNodes>")

        count=0
        offset = 0
        annotId = 245456
        annotz = []
        nodeIds = {}

        alldocs = list(self.data.values())
        random.shuffle(alldocs)
        for e in tqdm(alldocs):
            count+=1
            testjson  = e.doc.to_json()
            prevEnd = -1
            
            for sent in e.doc.sents:
                for token in sent:
                    prevEnd = createNode(token,e.doc,offset,xmlFile,prevEnd)\

                    start = offset + e.doc[token.i:token.i+1].start_char
                    try:
                        nodeIds[start]+=1
                    except KeyError:
                        nodeIds[start]=1
                    
                    
                    end = offset + e.doc[token.i:token.i+1].end_char
                    try:
                        nodeIds[end]+=1
                    except KeyError:
                        nodeIds[end]=1


            
            txtFile.write(testjson["text"] + "\n\n")
            


            for tok in testjson["tokens"]:
                tempToken = {}
                tempToken["category"] = tok["tag"]
                tempToken["kind"] = tok["dep"]
                tempToken["id"] = tok["id"]
                tempToken["head"] = tok["head"]
                
                annotz.append([annotId, "Token", offset + tok["start"], offset + tok["end"], tempToken, xmlFile])
                #createAnnotation(annotId, "Token", offset + tok["start"], offset + tok["end"], tempToken, xmlFile)
                annotId += 1
                    
                    
            for sent in testjson["sents"]:
                tempSent = {}
                annotz.append([annotId, "sentence", offset + sent["start"], offset + sent["end"], tempSent, xmlFile])
                #createAnnotation(annotId, "sentence", offset + sent["start"], offset + sent["end"], tempSent, xmlFile)
                annotId += 1 
                
                    
            # for index, row in e.tsv.iterrows():
            #     tempAnnot = {}
            #     tempAnnot["annotSet"] = row["annotSet"]
            #     tempAnnot["annotId"] = row["annotId"]
            #     tempAnnot["text"] = row["text"]
            #     if(type(row["other"]) == str):
            #         tempAnnot["other"] = row["other"]
            #     else:
            #         tempAnnot["other"] = "nothing"
                    
            #     annotz.append([annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile])    
            #     #createAnnotation(annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile)
            #     annotId += 1

            for x in e.doc._.meAnnotTrain:
                tempAnnot = {}
                try: 
                    tok = x["Quantity"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)

                    annotz.append([annotId, "MEval-Quantity", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["MeasuredEntity"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-MeasuredEntity", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["MeasuredProperty"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-MeasuredProperty", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["Qualifier"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-Qualifier", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass
            
                    
            for num in e.doc._.h0Number:
                temp= {}
                temp["text"]= num["text"]
                
                annotz.append([annotId, "h0Number", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0Number", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                    
            for num in e.doc._.h0Unit:
                temp= {}
                temp["text"]= num["text"]
                
                annotz.append([annotId, "h0Unit", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0Unit", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                    
            for num in e.doc._.h0MeasuredEntity:
                temp= {}
                temp["text"]= num["text"]
                
                annotz.append([annotId, "h0MeasuredEntity", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0MeasuredEntity", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                    
            #True Positives        
            for num in e.doc._.h0NumberTps:
                temp= {}
                temp["text"]= num["text"]
                
                annotz.append([annotId, "h0NumberTP", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0NumberTP", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                
            for num in e.doc._.h0UnitTps:
                temp= {}
                temp["text"]= num["text"]
                
                annotz.append([annotId, "h0UnitTP", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0UnitTP", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                    
            for num in e.doc._.h0MeasuredEntityTps:
                temp= {}
                temp["text"] = num["text"]
                
                annotz.append([annotId, "h0MeasuredEntityTP", offset + num["start"], offset + num["end"], temp, xmlFile])
                #createAnnotation(annotId, "h0MeasuredEntityTP", offset + num["start"], offset + num["end"], temp, xmlFile)
                annotId += 1 
                    
            offset += len(testjson["text"])
            if count > numberOfDocs:
                break
            
            
        xmlFile.write("\n</TextWithNodes>\n\n")    
            
        xmlFile.write("<AnnotationSet Name=\"Bens annots\">\n")

        for x in annotz:
            try:
                nodeIds[x[2]]+=1
                nodeIds[x[3]]+=1
                if(x[2]>=x[3]):
                    print("weird span: ({},{}) for {}".format(x[2],x[3],x))
                else:
                    createAnnotation(*x)
            except KeyError:
                print("unmatched NodeId for {}".format(x))
            
        xmlFile.write("</AnnotationSet>")
        xmlFile.write("</GateDocument>")    

        xmlFile.close()
        txtFile.close()

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
        
    


            






    