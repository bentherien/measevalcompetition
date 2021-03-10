import spacy
import src.common as common



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

def createAnnotation(ID, tpe, start, end, features, file):
    file.write(f"<Annotation Id=\"{ID}\" Type=\"{tpe}\" StartNode=\"{start}\" EndNode=\"{end}\">\n")
    for key in features.keys():
        createFeature(key,features[key],file)
    file.write("</Annotation>\n")
    
def createNode(token,doc,offset,file,prevEnd):
    txt = token.text
    txt = txt.replace("&","&amp;")

    
    txt = txt.replace("'","&apos;")
    txt = txt.replace("\"","&quot;")
    txt = txt.replace("<","&lt;")
    txt = txt.replace(">","&gt;")
    
    start = doc[token.i:token.i+1].start_char+offset
    end = offset+doc[token.i:token.i+1].end_char
    closestMul = round(start/200)
    
    if(start == prevEnd):
        if(start/200<=closestMul and end/200 >= closestMul):
            file.write("{}<Node id=\"{}\"/>\n".format(txt,end))
        else:
            file.write("{}<Node id=\"{}\"/>".format(txt,end))
        
    elif(start > prevEnd):
        if(start/200<=closestMul and end/200 >= closestMul):
            file.write(" <Node id=\"{}\"/>{}<Node id=\"{}\"/>\n".format(start,txt,end))
        else:
            file.write(" <Node id=\"{}\"/>{}<Node id=\"{}\"/>".format(start,txt,end))
    else:
        print("case Unhandled")
    
    return end
    


def createFeature(key, value, file):
    file.write(f"""<Feature>
  <Name className="java.lang.String">{key}</Name>
  <Value className="java.lang.String">{value}</Value>
</Feature>\n""")




def setupNLP(new = False):
    #prefix: − − ±
    # infix: – ±
    #suffix Rp

    if not new:
        #infixes = common.nlp.Defaults.infixes + (r'''=''',r'''~''',r'''•''',r'''∼''',r'''•''',r''']''',r''',''',)
        infixes = common.nlp.Defaults.infixes + (r'''=''',r'''~''',r'''•''',r'''∼''',r'''•''',r''']''',
        r''',''',r''':''',r'''%''',r'''‰''',r'''\\(''',r''')''',r'''→''',r'''\\+''',)
        infix_regex = spacy.util.compile_infix_regex(infixes)
        common.nlp.tokenizer.infix_finditer = infix_regex.finditer

        prefixes =common.nlp.Defaults.prefixes + (r'''=''',r'''~''',r'''•''',r'''∼''',r'''•''',)
        #prefixes = common.nlp.Defaults.prefixes + (r'''=''',r'''~''',r'''•''',r'''∼''',r'''•''',r'''[''',)
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        common.nlp.tokenizer.prefix_search = prefix_regex.search
    else:
        infixes = common.nlp.Defaults.infixes + [r'''[-=~•∼\]:%‰\()→+−±–,><≈]''']
        infix_regex = spacy.util.compile_infix_regex(infixes)
        common.nlp.tokenizer.infix_finditer = infix_regex.finditer

        prefixes = common.nlp.Defaults.prefixes + [r'''[-=~•∼±−≥><≤]''']
        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        common.nlp.tokenizer.prefix_search = prefix_regex.search

        suffixes = common.nlp.Defaults.suffixes + [r'''R{1}p{1}|R{1}s{1}|m{1}H{1}|R{1}R{1}h{1}|k{1}R{1}''',r'''\.''']
        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        common.nlp.tokenizer.suffix_search = suffix_regex.search

