"""
This file is dedicated to any external processing modules for spacy documents.

There are modules that should be a part of the spacy pipeline, but that 
cannot be added to the pipeline while following desirable SE principles
"""


from src.lib.helpers import findOffset, intersectSpan, getSentences, intersectSpanNum, intersectSpanSpan
from spacy.tokens import Doc
import math
import logging
import json
import src.common as common

def annotationCreation(doc,tsv):
    try:
        doc._.meAnnots
    except Exception: 
        Doc.set_extension("meAnnots", default = "def", force = True)

    doc._.meAnnots = {}
    count = common.count
    lookup = {tok.idx : tok.i for tok in doc}
    annotminmax={}
    #print("start")
    for index, row in tsv.iterrows():
        if(row["annotType"] == "Quantity"):
            count += 1
            #print(count,">",common.count+1)
            if count > common.count + 1:
                #set sentence values for each
                doc._.meAnnots[f"Annotation{count-1}"]["sentences"] = getSentences(annotminmax[f"offset{count-1}min"],annotminmax[f"offset{count-1}max"],doc)
        
        #get min and max offsets
        try:
            annotminmax[f"offset{count}max"] = max(annotminmax[f"offset{count}max"],row["endOffset"])
        except KeyError:
            annotminmax[f"offset{count}max"] = row["endOffset"]
                
        try:
            annotminmax[f"offset{count}min"] = min(annotminmax[f"offset{count}min"],row["startOffset"])
        except KeyError:
            annotminmax[f"offset{count}min"] = row["startOffset"]
            
        
        #doc._.meAnnots[f"Annotation{count}"]
        
        #check for creating a new dict
        try:
            if(type(doc._.meAnnots[f"Annotation{count}"]) == type(dict)):
                continue
        except KeyError:
            doc._.meAnnots[f"Annotation{count}"] = {}
        

        def getClosestMatch(start,end,lookup):
            print("in getClosestMatch(start,end,lookup)")
            newStart = None
            newEnd = None
            for x in lookup.keys():
                if x < start: 
                    newStart = x
                if x > end and newEnd == None: 
                    newEnd = x
                    return newStart, newEnd

            print(f"Error, nothing found for start:{start}, end:{end}, \nlookup:{lookup}")
            
            if newStart ==None and newEnd == None:
                return 0,list(lookup.keys())[-1]
            elif newStart == None and newEnd  != None:     
                return 0,newEnd
            elif newStart != None and newEnd  == None: 
                return newStart,list(lookup.keys())[-1]
            else:
                print("error occured in file execution should not have reached this point, componentsExternSpacy.py getClosestMatch(start,end,lookup)")

            return newStart, newEnd 

        error = False

        tempStart = row["startOffset"]
        lookupStart = None
        try:
            lookupStart = lookup[tempStart]
        except KeyError:
            try:
                tempStart = findOffset(row["startOffset"],doc.text)
                lookupStart = lookup[tempStart]
            except KeyError:
                tempStart, _ = getClosestMatch(row["startOffset"],row["endOffset"],lookup)
                lookupStart = lookup[tempStart]
                error = True
                    
                    

        tempEnd = row["endOffset"]
        lookupEnd = None
        try:
            lookupEnd = lookup[tempEnd]
        except KeyError:
            try:
                if(findOffset(row["endOffset"],doc.text) > list(lookup.keys())[-1]):
                    tempEnd = list(lookup.keys())[-1]
                    lookupEnd = lookup[tempEnd]+1
                else:
                    tempEnd = findOffset(row["endOffset"],doc.text)
                    lookupEnd = lookup[tempEnd]
            except KeyError:

                _, tempEnd = getClosestMatch(row["startOffset"],row["endOffset"],lookup)
                lookupEnd = lookup[tempEnd]
                error = True


        tempSpan = None
        if tempStart <= tempEnd:
            tempSpan = doc[lookupStart:lookupEnd]
            if len(tempSpan) > 25:
                print(row["annotType"])
                print(tempSpan)
        else:
            print("ERROR tempstart greater than temp end")

        if error and False:
            print("FindOffset method has created a key error ")
            print("Annotation Type: {}".format(row["annotType"]))
            print("Text + 10 on each side: \"{}\"".format(doc.text[max(0,row["startOffset"]-10):min(len(doc.text),row["endOffset"]+10)]))
            print("Gold text            : \"{}\"".format(row["text"]).encode().decode())
            print("Gold text from doc   : \"{}\"".format(doc.text[row["startOffset"]:row["endOffset"]]))
            print("compromise           : \"{}\"".format(doc.text[int(tempStart):int(tempEnd)]))
            print("corrected range     : ("+str(findOffset(row["startOffset"],doc.text))+","+str(findOffset(row["endOffset"],doc.text))+")")
            print("origrange: ("+str(row["startOffset"])+","+str(row["endOffset"])+")")
            print("TextSize: (0,"+str(len(doc.text))+")")
            print("Compromise range: ("+str(tempStart)+","+str(tempEnd)+")")
            print({k:lookup[k] for k in lookup.keys() if(k > row["startOffset"]-10 and k < row["endOffset"]+10)})            
        
        other = (json.loads(str(row["other"])) if str(row["other"]) != "nan" else {})
        other["annotID"] = count
        doc._.meAnnots[f"Annotation{count}"][row["annotType"]] = {"span":tempSpan,"other":other}
      
    doc._.meAnnots[f"Annotation{count}"]["sentences"] = getSentences(annotminmax[f"offset{count}min"],annotminmax[f"offset{count}max"],doc)
    common.count = count
    return doc   


def evaluate(exerpt):    
    try:
        doc._.h0NumberTps
        doc._.h0UnitTps
        doc._.h0MeasuredEntityTps
        doc._.h0NumberFps
        doc._.h0UnitFps
        doc._.h0MeasuredEntityFps
        doc._.h0MeasurementTps
    except Exception: 
        Doc.set_extension("h0NumberTps", default = "def", force = True)
        Doc.set_extension("h0UnitTps", default = "def", force = True)
        Doc.set_extension("h0MeasuredEntityTps", default = "def", force = True)
        Doc.set_extension("h0NumberFps", default = "def", force = True)
        Doc.set_extension("h0UnitFps", default = "def", force = True)
        Doc.set_extension("h0MeasuredEntityFps", default = "def", force = True)
        Doc.set_extension("h0MeasurementTps", default = "def", force = True)


    exerpt.doc._.h0NumberTps = []
    exerpt.doc._.h0UnitTps = []
    exerpt.doc._.h0MeasuredEntityTps = []
    exerpt.doc._.h0NumberFps = []
    exerpt.doc._.h0UnitFps = []
    exerpt.doc._.h0MeasuredEntityFps = []
    exerpt.doc._.h0MeasurementTps = []

    tempMeas = list(exerpt.measurements.values())
    for meas in exerpt.doc._.h0Measurements:
        num = meas["Number"]
        unit = meas["Unit"]
        me = meas["MeasuredEntity"]
        
        
        count = 0
        for m in tempMeas:
            try:

                if intersectSpan(num["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"]):
                    exerpt.doc._.h0NumberTps.append(num)

                if intersectSpan(unit["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"]):
                    exerpt.doc._.h0UnitTps.append(unit)


                if(intersectSpan(num["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"]) and intersectSpan(unit["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"])):
                    
                    if(intersectSpan(me["span"],m["MeasuredEntity"]["startOffset"],m["MeasuredEntity"]["endOffset"]) or
                    intersectSpanNum(me["start"],me["end"],m["MeasuredEntity"]["startOffset"],m["MeasuredEntity"]["endOffset"])):
                        exerpt.doc._.h0MeasuredEntityTps.append(num)
                        exerpt.doc._.h0MeasurementTps.append(meas)
                    else:
                        r = dict(meas)
                        del r["MeasuredEntity"]
                        exerpt.doc._.h0MeasurementTps.append(r)

                    #remove once counted to prevent double counting 
                    tempMeas.pop(count)
            except KeyError:
                pass#print("No quantity")
            count+=1


def getSplit(exerpt):
    doc = exerpt.doc    

    try:
        doc._.meAnnotTest
        doc._.meAnnotTrain
    except Exception: 
        Doc.set_extension("meAnnotTest", default = "def", force = True)
        Doc.set_extension("meAnnotTrain", default = "def", force = True)

    temp = list(exerpt.doc._.meAnnots.values())

    exerpt.doc._.meAnnotTest = [temp[0]]
    if len(temp) > 1:
        exerpt.doc._.meAnnotTrain = temp[1:]
    else: 
        exerpt.doc._.meAnnotTrain = []


def getCRFSplit(exerpt):
    doc = exerpt.doc    

    try:
        doc._.crfTest 
        doc._.crfTrain
    except Exception: 
        Doc.set_extension("crfTest", default = "def", force = True)
        Doc.set_extension("crfTrain", default = "def", force = True)

    doc._.crfTest = {"raw": [], "obj": []}
    doc._.crfTrain = {"raw": [], "obj": []}

    for x in doc._.meAnnotTest:
        quant = x["Quantity"]
        for sent in x["sentences"]:
            if intersectSpanSpan(quant,sent):
                rawSent = []
                objSent = [] 
                quantIds = [y.i for y in quant]
                for token in sent:
                    if token.i in quantIds:
                        rawSent.append((token.text, token.tag_,"Q"))
                        objSent.append((token, token.tag_,"Q"))
                    else:
                        rawSent.append((token.text, token.tag_,"O"))
                        objSent.append((token, token.tag_,"O"))


                doc._.crfTest["raw"].append(rawSent)  
                doc._.crfTest["obj"].append(objSent)


    for x in doc._.meAnnotTrain:
        quant = x["Quantity"]
        for sent in x["sentences"]:
            if intersectSpanSpan(quant,sent):
                rawSent = []
                objSent = [] 
                quantIds = [y.i for y in quant]
                for token in sent:
                    if token.i in quantIds:
                        rawSent.append((token.text, token.tag_,"Q"))
                        objSent.append((token, token.tag_,"Q"))
                    else:
                        rawSent.append((token.text, token.tag_,"O"))
                        objSent.append((token, token.tag_,"O"))


                doc._.crfTrain["raw"].append(rawSent)  
                doc._.crfTrain["obj"].append(objSent)






    

    

