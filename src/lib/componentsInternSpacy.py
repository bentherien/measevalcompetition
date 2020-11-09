from spacy.matcher import Matcher
from spacy.tokens import Doc
import src.common as common




def gazetteer(doc):
    """
    Description: A gazetteer 
    Doc.set_extension("unit", default = "def", force = True)
    """
    try:
        doc._.unit
    except Exception: 
        Doc.set_extension("unit", default = "def", force = True)

    def customMatcher():
        """
        Description: matcher giving the most recall so far
        """
        matchList = open("gazetteers/combined_measurements.lst","r",encoding="utf-8").read().split("\n")
        matcher = Matcher(common.nlp.vocab)
        pattern = []
        for word in matchList: 
            pattern.append([{"TAG": {"REGEX": "^[CD]"}},{"LOWER": word.lower(),"TAG": {"REGEX": "^[NN|NNP|NNPS|NNS]"}}])
            pattern.append([{"ENT_TYPE": {"IN": ["CARDINAL", "MONEY", "ORDINAL", "PERCENT", "DATE", "TIME", "QUANTITY"]},
                            "TAG":{"REGEX": "^[DT]"},"op": "!"},{"LOWER": word.lower(),"TAG": {"REGEX": "^[NN|NNP|NNPS|NNS]"}}])
            pattern.append([{"LIKE_NUM": True},{"LOWER": word.lower(),"TAG": {"REGEX": "^[NN|NNP|NNPS|NNS]"}}])
            #pattern.append([{"LOWER": word.lower(),"TAG": {"REGEX": "^[NN|NNP|NNPS|NNS]"}}])
            
        matcher.add("Unit", None, *pattern)
            
        return matcher



    matcher = customMatcher()
    matches = matcher(doc)
    doc._.unit = []
    for match_id, start, end in matches:
        tempSpan = doc[start:end]
        doc._.unit.append({'start': tempSpan.start_char, 'end': tempSpan.end_char, 'label': 'UNIT', 'text' : doc[start:end]})
    return doc



#pipeline component H0






def h0(doc):
    """
    Description: Pipeline component for hypothesis 0

    Doc.set_extension("h0Number", default = "def", force = True)
    Doc.set_extension("h0Unit", default = "def", force = True)
    Doc.set_extension("h0MeasuredEntity", default = "def", force = True)
    Doc.set_extension("h0Measurements", default = "def", force = True)
    """
    try:
        doc._.h0Number
        doc._.h0Unit
        doc._.h0MeasuredEntity
        doc._.h0Measurements
    except Exception: 
        Doc.set_extension("h0Number", default = "def", force = True)
        Doc.set_extension("h0Unit", default = "def", force = True)
        Doc.set_extension("h0MeasuredEntity", default = "def", force = True)
        Doc.set_extension("h0Measurements", default = "def", force = True)

    def numberMatcher():
        """
        Description: matcher giving the most recall so far
        """
        matcher = Matcher(common.nlp.vocab)
        pattern = []
        pattern.append([{"LIKE_NUM": True}])
        pattern.append([{"ENT_TYPE": {"IN": ents}}])
        matcher.add("h0Number", None, *pattern)
            
        return matcher


    ents = ["CARDINAL", "MONEY", "PERCENT", "DATE", "TIME", "QUANTITY"]

    matcher = numberMatcher()
    matches = matcher(doc)
    doc._.h0Number = []
    doc._.h0Unit = []
    doc._.h0MeasuredEntity = []
    doc._.h0Measurements = []
    for match_id, start, end in matches:
        
        tempSpan = doc[start:end]
        tempTok = doc[start]
        tempNum = {
            'start': tempSpan.start_char, 
            'end': tempSpan.end_char, 
            'label': 'h0Number', 
            'text' : tempTok.text,
            'span' : tempSpan,
            's' : start,
            'e' : end
        }
        
        doc._.h0Number.append(tempNum)
        
        tempHead = tempTok.head
        spanHead = doc[tempHead.i:tempHead.i+1]
        tempUnit = {
            'start': spanHead.start_char, 
            'end': spanHead.end_char, 
            'label': 'h0Unit', 
            'text' : tempHead.text,
            'span' : spanHead,
            's' : tempHead.i,
            'e' : tempHead.i+1
        }
        
        doc._.h0Unit.append(tempUnit)
        
        tempHeadHead = None
        spanHeadHead = None
        if tempHead.dep_ == "pobj":
            tempHeadHead = tempTok.head.head.head
            spanHeadHead = doc[tempHeadHead.i:tempHeadHead.i+1]
        else:
            tempHeadHead = tempTok.head.head
            spanHeadHead = doc[tempHeadHead.i:tempHeadHead.i+1]
            
        
        tempME = {
            'start': spanHeadHead.start_char, 
            'end': spanHeadHead.end_char, 
            'label': 'h0MeasuredEntity', 
            'text' : tempHeadHead.text,
            'span' : spanHeadHead,
            's' : tempHeadHead.i,
            'e' : tempHeadHead.i+1
        }
        
        doc._.h0MeasuredEntity.append(tempME)
        
        doc._.h0Measurements.append({
            "Number" : tempNum,
            "Unit" : tempUnit,
            "MeasuredEntity": tempME
        })
        
        
    return doc

