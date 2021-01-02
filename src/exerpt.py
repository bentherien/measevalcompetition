import json
import math
import os
import networkx as nx

from src.graph import Graph
import src.common as common
from src.model.load_util import Sample
from src.lib.componentsExternSpacy import annotationCreation, evaluate, getSplit, getCRFSplit
from src.lib.helpers import intersectSpan,intersectSpanSpan

class Exerpt:
    """
    Class exerpt
    Description: a simple class to contain data for the measeval competition

    self.name : the measeval given name associated to the document
    self.txt : the raw text of the document
    self.ann : the brat annotations of the document(deprecated)
    self.tsv : a pandas dataframe containing all the tab seperated value data
    self.grobid : grobid quantities json output for quantity detection
    self.doc : The spacy doc generated from processing on this particular document
    """
    
    def __init__(self, name, txt, ann, tsv, grobid):
        self.name = name
        self.txt = txt
        self.ann = ann
        self.tsv = tsv
        self.grobid = grobid
        self.context = False
        self.posTag = False
        self.doc = common.nlp(self.txt)
        self.annotationsToDict()
        annotationCreation(self.doc,self.tsv)
        evaluate(self)
        getSplit(self)
        #getCRFSplit(self)

    def initGraphs(self):

        allsents = []
        for annot in self.doc._.meAnnots.values():
            if len(annot["sentences"]) == 1 :
                for sent in annot["sentences"]:
                    append = True
                    for s in allsents:
                        if sent.start == s.start:
                            append = False
                    if append:
                        allsents.append(sent)

        self.graphs = []
        for sentence in allsents:
            self.graphs.append(Graph(sentence))
    
    

        
    def getContext(self, count = 3):
        if self.context == False: 
            beforeContext = []
            afterContext = []
            self.context = True
            for start,end in zip(self.tsv["startOffset"].values, self.tsv["endOffset"].values ):#self.tsv.iterrows():
                a = getIndexSeperated(exerpt.txt[0],end,count = count)
                b = getIndexSeperated(exerpt.txt[0],start,count = count,forward = False)
                beforeContext.append(self.txt[0][b:start])
                afterContext.append(self.txt[0][end:a])
            self.tsv.insert(3,"beforeContext", beforeContext)
            self.tsv.insert(3,"afterContext", afterContext)
            
    def annotationsToDict(self):
        self.measurements = {}
        count = 0
        for index, row in self.tsv.iterrows():
            if(row["annotType"] == "Quantity"):
                count+=1
                
        
            #check for creating a new dict
            try:
                if(type(self.measurements[f"Annotation{count}"]) == type(dict)):
                    continue
            except KeyError:
                self.measurements[f"Annotation{count}"] = {}

            
            self.measurements[f"Annotation{count}"][row["annotType"]] = {
                "startOffset" : row["startOffset"],
                "endOffset" : row["endOffset"],
                "annotSet" : row["annotSet"],
                "annotType" : row["annotType"],
                "annotId" : row["annotId"],
                "text" : row["text"],
                "other" : row["other"],
            }

            
    def getPosTag(self):
        if self.context and self.posTag == False: 
            bTag = []
            aTag = []
            self.posTag = True
            for before, after in zip(self.tsv["beforeContext"].values,self.tsv["afterContext"].values):
                bTag.append(nltk.pos_tag(word_tokenize(before)))
                aTag.append(nltk.pos_tag(word_tokenize(after)))
            self.tsv.insert(3,"beforeTag", bTag)
            self.tsv.insert(3,"afterTag", aTag)


    def getAscii(self, filepath):
        """
        Dumps and Ascii output of the current document to the specified filepath
        """
        fileObj = open(os.path.join(filepath,self.name+".ascii.txt"),"w",encoding="utf-8")

        text = ""
        count = 0
        counts = []
        dobreak = False
        for sent in self.doc.sents: 
            count+=1
            if dobreak == False and count > 1:
                pass

            for meas in self.measurements.values():
                dobreak=False
                for annot in meas.values():
                    if(intersectSpan(sent,annot["startOffset"],annot["endOffset"])):
                        counts.append(count)
                        text += sent.text
                        dobreak=True
                        break
                        
                if dobreak:
                    break
                
            
        fileObj.write(f"Document {self.name}:\n\n")
        fileObj.write(text+"\n\n")
        
        fileObj.write("Gold Annotations:\n\n")
        fp = open(f"data-merged/tsv/{self.name}.tsv","r",encoding="utf-8")
        fileObj.write(fp.read())
        fp.close()
        
        fileObj.write("\n\nHypothesis 0 annotations:\n")
        count =0
        
        for meas in self.doc._.h0Measurements:
            count+=1
            fileObj.write(f"\nMeasurement {count}\n")
            num = meas["Number"]
            unit = meas["Unit"]
            me = meas["MeasuredEntity"]
            fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
            fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
            fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))

            
        fileObj.write("\n\nCorrect Hypothesis 0 annotations:\n")
        if len(self.doc._.h0MeasurementTps) > 0:
            

            for meas in self.doc._.h0MeasurementTps:
                count+=1
                fileObj.write(f"\nMeasurement {count}\n")
                if len(meas) == 3:
                    num = meas["Number"]
                    unit = meas["Unit"]
                    me = meas["MeasuredEntity"]
                    fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
                    fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
                    fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))
                elif len(meas) == 2:
                    num = meas["Number"]
                    unit = meas["Unit"]
                    fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
                    fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
        else: 
            fileObj.write("\nNONE\n")

        
        
        
        fileObj.write("\n\nDependencies:\n")
        count =0
        for sentence in self.doc.sents:
            count+=1
            if count in counts:
                print(f"Sentence {count}: \n\n", sentence, "\n",file = fileObj)
                print("governor".ljust(15), "dependency".ljust(15), "token".ljust(15),file = fileObj)
                fileObj.write("-----------------------------------------\n")
                for token in sentence:
                    print(token.head.text.ljust(15),token.dep_.ljust(15), token.text.ljust(15),file = fileObj)
                print("\n\n",file = fileObj)

        fileObj.close()

    def getAsciiConstituent(self, filepath, constituency = False):
        fileObj = open(os.path.join(filepath,self.name+".ascii.txt"),"w",encoding="utf-8")

        text = ""
        count = 0
        counts = []
        dobreak = False
        for sent in self.doc.sents: 
            count+=1
            if dobreak == False and count > 1:
                pass
            
            for meas in self.measurements.values():
                dobreak=False
                for annot in meas.values():
                    if(intersectSpan(sent,annot["startOffset"],annot["endOffset"])):
                        counts.append(count)
                        text += sent.text
                        dobreak=True
                        break
                        
                if dobreak:
                    break
                
            
        fileObj.write(f"Document {self.name}:\n\n")
        fileObj.write(text+"\n\n")
        
        fileObj.write("Gold Annotations:\n\n")
        fp = open(f"data-merged/tsv/{self.name}.tsv","r",encoding="utf-8")
        fileObj.write(fp.read())
        fp.close()

        print("\n\nGovernor".ljust(15), "Dependency".ljust(15), "Quantity".ljust(15),file = fileObj)
        fileObj.write("-----------------------------------------\n")
        for x in self.doc._.meAnnots.values():
            try:
                if(x["Quantity"][len(x["Quantity"])-1].text in [")",".",",",":","/",";","-"]):
                    temp = x["Quantity"][len(x["Quantity"])-2]
                    print(temp.head.text.ljust(15), temp.dep_.ljust(15), temp.text.ljust(15),file = fileObj)
                else: 
                    temp = x["Quantity"][len(x["Quantity"])-1]
                    print(temp.head.text.ljust(15), temp.dep_.ljust(15), temp.text.ljust(15),file = fileObj)
            except TypeError: 
                continue
        
        fileObj.write("\n\nHypothesis 0 annotations:\n")
        count =0
        
        for meas in self.doc._.h0Measurements:
            count+=1
            fileObj.write(f"\nMeasurement {count}\n")
            num = meas["Number"]
            unit = meas["Unit"]
            me = meas["MeasuredEntity"]
            fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
            fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
            fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))

            
        fileObj.write("\n\nCorrect Hypothesis 0 annotations:\n")
        if len(self.doc._.h0MeasurementTps) > 0:
            

            for meas in self.doc._.h0MeasurementTps:
                count+=1
                fileObj.write(f"\nMeasurement {count}\n")
                if len(meas) == 3:
                    num = meas["Number"]
                    unit = meas["Unit"]
                    me = meas["MeasuredEntity"]
                    fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
                    fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
                    fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))
                elif len(meas) == 2:
                    num = meas["Number"]
                    unit = meas["Unit"]
                    fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
                    fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
        else: 
            fileObj.write("\nNONE\n")
    
        fileObj.write("\n\nDependencies:\n")
        count =0
        for sentence in self.doc.sents:
            count+=1
            if count in counts:
                print(f"Sentence {count}: \n\n", sentence, "\n",file = fileObj)
                if constituency:
                    pp = pprint.PrettyPrinter(indent=3, width=80, depth=5,  stream = fileObj)
                    print("\nConstituency parse: \n\n",file = fileObj)
                    print(sentencself._.parse_string,file = fileObj)
                    #pp.pprint(sentencself._.parse_string)
                    print("\n\n",file = fileObj)
                print("governor".ljust(15), "dependency".ljust(15), "token".ljust(15),file = fileObj)
                fileObj.write("-----------------------------------------\n")
                for token in sentence:
                    print(token.head.text.ljust(15),token.dep_.ljust(15), token.text.ljust(15),file = fileObj)
                print("\n\n",file = fileObj)

        fileObj.close()

    def getGateJson(self, filepath):

        testjson  = self.doc.to_json()


        twitjson = {"text": testjson["text"],"entities":{}}

        for tok in testjson["tokens"]:
            tempToken = {}
            tempToken["indices"] = [tok["start"],tok["end"]] 
            tempToken["category"] = tok["tag"]
            tempToken["kind"] = tok["dep"]
            tempToken["id"] = tok["id"]
            tempToken["head"] = tok["head"]
            
                
            try:
                twitjson["entities"]["Token"].append(tempToken)
            except KeyError:
                twitjson["entities"]["Token"] = [tempToken] 

        for ent in testjson["ents"]:
            tempEnt = {}
            tempEnt["indices"] = [ent["start"],ent["end"]] 
            try:
                twitjson["entities"][ent["label"]].append(tempEnt)
            except KeyError:
                twitjson["entities"][ent["label"]] = [tempEnt]
                
                
        for sent in self.doc.sents:
            for tok in sent: 
                tempEnt = {}
                if tok.dep_ == "root":
                    tempEnt["args"] = ["",tok.text]
                else:
                    tempEnt["args"] = [tok.head.text,tok.text]
                    
                tempEnt["kind"] = tok.dep_
                
                mn = min(self.doc[tok.head.i:tok.head.i+1].start_char,self.doc[tok.i:tok.i+1].start_char)
                mx = max(self.doc[tok.head.i:tok.head.i+1].end_char,self.doc[tok.i:tok.i+1].end_char)
                
                tempEnt["indices"] = [mn,mx]
            
                try:
                    twitjson["entities"]["NickDependency"].append(tempEnt)
                except KeyError:
                    twitjson["entities"]["NickDependency"] = [tempEnt]
                
                
        #     for unit in doc.doc._.unit:
        #         tempUnit = {}
        #         tempUnit["indices"] = [int(unit["start"]),int(unit["end"])]
        #         tempUnit["text"]= unit["text"].text
        #         try:
        #             twitjson["entities"]["unit"].append(tempUnit)
        #         except KeyError:
        #             twitjson["entities"]["unit"] = [tempUnit]

                
        for sent in testjson["sents"]:
            tempSent = {}
            tempSent["indices"] = [sent["start"],sent["end"]] 
            try:
                twitjson["entities"]["sentence"].append(tempSent)
            except KeyError:
                twitjson["entities"]["sentence"] = [tempSent] 
                
        for index, row in self.tsv.iterrows():
            tempAnnot = {}
            tempAnnot["indices"] = [row["startOffset"],row["endOffset"]] 
            tempAnnot["annotSet"] = row["annotSet"]
            tempAnnot["annotId"] = row["annotId"]
            tempAnnot["text"] = row["text"]
            if(type(row["other"]) == str):
                tempAnnot["other"] = row["other"]
            else:
                tempAnnot["other"] = "nothing"
                
            try:
                twitjson["entities"]["MEval-"+row["annotType"]].append(tempAnnot)
            except KeyError:
                twitjson["entities"]["MEval-"+row["annotType"]] = [tempAnnot] 
                
        #      doc._.h0Number = []
        #     doc._.h0Unit = []
        #     doc._.h0MeasuredEntity = []
                
                
        for num in self.doc._.h0Number:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0Number"].append(temp)
            except KeyError:
                twitjson["entities"]["h0Number"] = [temp]
                
        for num in self.doc._.h0Unit:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0Unit"].append(temp)
            except KeyError:
                twitjson["entities"]["h0Unit"] = [temp]
                
        for num in self.doc._.h0MeasuredEntity:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0MeasuredEntity"].append(temp)
            except KeyError:
                twitjson["entities"]["h0MeasuredEntity"] = [temp]
                
        #True Positives        
        for num in self.doc._.h0NumberTps:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0NumberTP"].append(temp)
            except KeyError:
                twitjson["entities"]["h0NumberTP"] = [temp]
                
        for num in self.doc._.h0UnitTps:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0UnitTP"].append(temp)
            except KeyError:
                twitjson["entities"]["h0UnitTP"] = [temp]
                
        for num in self.doc._.h0MeasuredEntityTps:
            temp= {}
            temp["indices"] = [int(num["start"]),int(num["end"])]
            temp["text"]= num["text"]
            try:
                twitjson["entities"]["h0MeasuredEntityTP"].append(temp)
            except KeyError:
                twitjson["entities"]["h0MeasuredEntityTP"] = [temp]



        json.dump(twitjson, open(os.path.join(filepath,f'{self.name}.json'),"w"), indent=3)


