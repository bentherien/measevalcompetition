"""
Author: Benjamin Therien

This file is dedicated to any external processing modules for spacy documents.

There are modules that should be a part of the spacy pipeline, but that 
cannot be added to the pipeline while following desirable SE principles
"""


from src.lib.helpers import intersectSpan, getSentences, intersectSpanNum, intersectSpanSpan
from spacy.tokens import Doc
import math
import logging
import json
import src.common as common

def annotationCreation(doc,tsv):
    """
    Description: Method used to create token level annotations under the meAnnots SpaCy 
    extension based on spans provided from the Measeval Tsv format

    Input
        doc : The document we wish to annotate with the tsv file passed.
        tsv : Pandas dataframe containing annotation contents from a Measeval tsv file.
    Output: 
        doc: The modified document.
    """

    def findOffset(offset, text):
        '''
        Corrects annotations whose offsets are incorrect
        '''
        #check for end of document
        if offset == len(text):
            return offset

        try:
            #find correct start offset
            if(text[offset] != " " and text[offset+1] != " "  and text[offset-1] == " "):
                return offset
                #no change
            elif(text[offset] != " " and text[offset+1] == "."):
                return offset
            elif(text[offset] != " " and text[offset+1] == " "):
                if(text[offset-1] == " "):
                #case where the word is one char
                    return offset
                    #no change
                else:
                    return offset+2
                    #skip ahead 2
            elif(text[offset] != " " and text[offset-2] == " "):
                return offset-1
            elif(text[offset] != " " and text[offset-3] == " "):
                return offset-2
            elif(text[offset] == " "):
                return offset + 1
            else:
                """
                print("error, unhandled case in findOffset()")
                print("offset", offset)
                print("offset", text[offset])
                print("offset +-10:", text[max(0, offset-10):min(len(text),offset+10)])
                """
                return offset
        except IndexError: 
            return offset

    def getClosestMatch(start,end,lookup):
        """
        Retrieves the closest match based on the lookup map
        """
        newStart = None
        newEnd = None
        for x in lookup.keys():
            if x < start: 
                newStart = x
            if x > end and newEnd == None: 
                newEnd = x

                if newStart != None:
                    return newStart, newEnd
                else:
                    break
        
        if newStart == None and newEnd == None:
            return 0,list(lookup.keys())[-1]
        elif newStart == None:     
            return 0,newEnd
        elif newEnd  == None: 
            return newStart,list(lookup.keys())[-1]
        else:
            print("error occured in file execution should not have reached this \
            point, componentsExternSpacy.py getClosestMatch(start,end,lookup)")

        return newStart, newEnd 

    try:
        doc._.meAnnots
    except Exception: 
        Doc.set_extension("meAnnots", default = {}, force = True)

    #A dict mapping: character offset : token offset
    lookup = {tok.idx : tok.i for tok in doc}

    count = common.count
    annotminmax={}
    for index, row in tsv.iterrows():
        if(row["annotType"] == "Quantity"):
            count += 1
            #print(count,">",common.count+1)
            if count > common.count + 1:
                #set sentence values for each complete data point as we iterate over them
                doc._.meAnnots[f"Annotation{count-1}"]["sentences"] = getSentences(annotminmax[f"offset{count-1}min"],annotminmax[f"offset{count-1}max"],doc)
        
        #Store the minimum and maximum offset values in order to retrieve the sentence offset later
        try:
            annotminmax[f"offset{count}max"] = max(annotminmax[f"offset{count}max"],row["endOffset"])
        except KeyError:
            annotminmax[f"offset{count}max"] = row["endOffset"]
                
        try:
            annotminmax[f"offset{count}min"] = min(annotminmax[f"offset{count}min"],row["startOffset"])
        except KeyError:
            annotminmax[f"offset{count}min"] = row["startOffset"]
            
        
        #Create a new dictionary if needed
        try:
            if(type(doc._.meAnnots[f"Annotation{count}"]) == type(dict)):
                pass
        except KeyError:
            doc._.meAnnots[f"Annotation{count}"] = {}
        

        
        error = False


        #Determine the token offset of the start of the span
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
                    
                    

        #Determine the token offset of the end of the span
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

        if tempStart > tempEnd:
            _, tempEnd = getClosestMatch(row["startOffset"],row["endOffset"],lookup)
            lookupStart,lookupEnd = lookup[tempStart],lookup[tempEnd]

        tempSpan = None
        if tempStart <= tempEnd:
            tempSpan = doc[lookupStart:lookupEnd]
            if len(tempSpan) > 25:
                print(row["annotType"])
                print(tempSpan)
        else:
            print("Gold|{}|".format(row["text"]))
            print(tempStart,tempEnd)
            print("ERROR tempstart greater than temp end")


        if abs(len(tempSpan.text) - len(row["text"])) > 0:
            # print("Cust|{}|".format(tempSpan.text))
            # print("Gold|{}|".format(row["text"]))
            try:
                common.c[len(tempSpan.text) - len(row["text"])] += 1
            except KeyError:
                common.c[len(tempSpan.text) - len(row["text"])] = 1

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
        

        #add the new annotation to meAnnots
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







    

    

