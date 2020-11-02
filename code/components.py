from code.helpers import findOffset, intersectSpan, getSentences
import math

def annotationCreation(doc):
    global LATEST_tsv
    doc._.meAnnots = {}
    count = 0
    lookup = {tok.idx : tok.i for tok in doc}
    annotminmax={}
    for index, row in LATEST_tsv.iterrows():
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
        
        try:
            tempSpan = doc[lookup[findOffset(row["startOffset"],doc.text)]:lookup[findOffset(row["endOffset"],doc.text)]]
        except KeyError:
            print("FindOffset method has created a key error ")
            print("origrange: (",row["startOffset"],",",row["endOffset"],")")
            print("range: (",findOffset(row["startOffset"],doc.text),",",findOffset(row["endOffset"],doc.text),")")
            print(lookup)
            
        doc._.meAnnots[f"Annotation{count}"][row["annotType"]] = tempSpan
      
    doc._.meAnnots[f"Annotation{count}"]["sentences"] = getSentences(annotminmax[f"offset{count}min"],annotminmax[f"offset{count}max"],doc)
        
    return doc        