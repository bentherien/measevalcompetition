import logging
import time
from tqdm import tqdm
import pandas as pd
import os
from src.exerpt import Exerpt
import json
import spacy
from src.lib.helpers import createAnnotation, createFeature, createNode

class ExerptController:
    """
    Description: Class containing all exerpts that encapsulates all the logic needed to deal with them 
    """


    def __init__(self, filepath):
        if(os.path.isdir(filepath)):
            self.data = self.readData(filepath)
        else:
            self.data = {}
        
    


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
        for e in tqdm(self.data.values()):
            count+=1
            testjson  = e.doc.to_json()
            prevEnd = -1
            
            for sent in e.doc.sents:
                for token in sent:
                    prevEnd = createNode(token,e.doc,offset,xmlFile,prevEnd)


            
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
                
                    
            for index, row in e.tsv.iterrows():
                tempAnnot = {}
                tempAnnot["annotSet"] = row["annotSet"]
                tempAnnot["annotId"] = row["annotId"]
                tempAnnot["text"] = row["text"]
                if(type(row["other"]) == str):
                    tempAnnot["other"] = row["other"]
                else:
                    tempAnnot["other"] = "nothing"
                    
                annotz.append([annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile])    
                #createAnnotation(annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile)
                annotId += 1
                    
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
            createAnnotation(*x)
            
        xmlFile.write("</AnnotationSet>")
        xmlFile.write("</GateDocument>")    

        xmlFile.close()
        txtFile.close()
            






    