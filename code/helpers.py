import spacy


def findOffset(offset, text):
    '''
    Corrects annotations whose offsets are not correct
    '''
    try:
        #find correct start offset
        if(text[offset] != " " and text[offset+1] != " "):
            return offset
            #no change
        elif(text[offset] != " " and text[offset+1] == " "):
            if(text[offset-1] == " "):
            #case where the word is one char
                return offset
                #no change
            else:
                return offset+2
                #skip ahead 2
        elif(text[offset] == " "):
            return offset + 1
        else:
            print("error, unhandled case in findOffset()")
            print("offset", offset)
            print("text", text)
            return offset
    except IndexError: 
        return offset

def intersectSpanSpan(span1, span2):
    if (type(span1) != spacy.tokens.Span) or (type(span2) != spacy.tokens.Span):
        return False
    
    s1 = span1.start_char
    e1 = span1.end_char
    s2 = span2.start_char
    e2 = span2.end_char
    
    return (s2 < s1 and s1 < e2 ) or\
(s2 < e1 and e1 < e2) or\
(s2 <= s1 and e1 <= e2) or\
(s1 <= s2 and e2 <= e1)

def intersectSpanNum(s1,e1,s2,e2):
    return (s2 < s1 and s1 < e2 ) or\
(s2 < e1 and e1 < e2) or\
(s2 <= s1 and e1 <= e2) or\
(s1 <= s2 and e2 <= e1)


def intersectSpan(span1,s2,e2):
    if type(span1) != spacy.tokens.Span:
        return False
    
    s1 = span1.start_char
    e1 = span1.end_char
    
    return (s2 < s1 and s1 < e2 ) or\
(s2 < e1 and e1 < e2) or\
(s2 <= s1 and e1 <= e2) or\
(s1 <= s2 and e2 <= e1)


def getSentences(omin,omax,doc):
    sents = []
    for sent in doc.sents:
        if intersectSpan(sent,omin,omax):
            sents.append(sent)
            
    return sents