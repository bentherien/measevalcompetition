"""
This file is dedicated to any external processing modules for spacy documents.

There are modules that should be a part of the spacy pipeline, but that 
cannot be added to the pipeline while following desirable SE principles
"""


from src.lib.helpers import findOffset, intersectSpan, getSentences, intersectSpanNum
from spacy.tokens import Doc
import math

def annotationCreation(doc,tsv):
    try:
        doc._.meAnnots
    except Exception: 
        Doc.set_extension("meAnnots", default = "def", force = True)

    doc._.meAnnots = {}
    count = 0
    lookup = {tok.idx : tok.i for tok in doc}
    annotminmax={}
    for index, row in tsv.iterrows():
        if(row["annotType"] == "Quantity"):
            count+=1
            if count > 1:
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
        
        tempSpan = None

        def getClosestMatch(start,end,lookup):
            newStart = None
            newEnd = None
            for x in lookup.keys():
                if x < start: 
                    newStart = x
                if x > end and newEnd == None: 
                    newEnd = x
                    return newStart, newEnd
            
            if newStart ==None and newEnd == None:
                print("error occured in file componentsExternSpacy.py getClosestMatch(start,end,lookup) returning {},{}".format(0,list(lookup.keys())[-1]))
                return 0,list(lookup.keys())[-1]
            elif newStart == None and newEnd  != None: 
                print("error occured in file componentsExternSpacy.py getClosestMatch(start,end,lookup) returning {},{}".format(0,newEnd))
                return 0,newEnd
            elif newStart != None and newEnd  == None: 
                print("error occured in file componentsExternSpacy.py getClosestMatch(start,end,lookup) returning {},{}".format(newStart,list(lookup.keys())[-1]))
                return newStart,list(lookup.keys())[-1]
            else:
                print("error occured in file execution should not have reached this point, componentsExternSpacy.py getClosestMatch(start,end,lookup)")

            return newStart, newEnd 

        try:
            tempSpan = doc[lookup[findOffset(row["startOffset"],doc.text)]:lookup[findOffset(row["endOffset"],doc.text)]]
        except KeyError:
            newStart, newEnd = getClosestMatch(row["startOffset"],row["endOffset"],lookup)
            print("new start {}, {}".format(newStart,newEnd))
            tempSpan = doc[lookup[newStart]:lookup[newEnd]]
            print("FindOffset method has created a key error ")
            print("Text + 5 on each side: \"{}\"".format(doc.text[max(0,row["startOffset"]-5):min(len(doc.text),row["endOffset"]+5)]))
            print("Gold text            : \"{}\"".format(row["text"]))
            print("new start {}, {}".format(newStart,newEnd))
            print("compromise           : \"{}\"".format(doc.text[int(newStart):int(newEnd)]))
            print("origrange: (",row["startOffset"],",",row["endOffset"],")")
            print("range: (",findOffset(row["startOffset"],doc.text),",",findOffset(row["endOffset"],doc.text),")")
            print({k:lookup[k] for k in lookup.keys() if(k > row["startOffset"]-5 and k < row["endOffset"]+5)})
            
            
            
            
        doc._.meAnnots[f"Annotation{count}"][row["annotType"]] = tempSpan
      
    doc._.meAnnots[f"Annotation{count}"]["sentences"] = getSentences(annotminmax[f"offset{count}min"],annotminmax[f"offset{count}max"],doc)  
            
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
                if(intersectSpan(num["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"]) and intersectSpan(unit["span"],m["Quantity"]["startOffset"],m["Quantity"]["endOffset"])):
                    exerpt.doc._.h0NumberTps.append(num)
                    exerpt.doc._.h0UnitTps.append(unit)
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