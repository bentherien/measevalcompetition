import src.exerptController
import networkx as nx



def getGraphPaths(sentences, sourceSpan, targetSpan, getSrc, getTrg, reverse=False, pos=False, docName = ""):
    nonePassed = False
    
    if(type(sentences) == type(None)):
        print("sentences passed to getGraphPaths() as None")
        nonePassed = True
        
    if(type(sourceSpan) == type(None)):
        print("quantity passed to getGraphPaths() as None")
        nonePassed = True
        
    if(type(targetSpan) == type(None)):
        print("measuredProperty passed to getGraphPaths() as None")
        nonePassed = True
        
    if(nonePassed):
        return None, None
        
    if len(sentences) != 1:
        #cant handle more than one sentece so break
        #print("more than one sentence")
        return None, None
    
    edges = []
    for sent in sentences:
        for token in sent:
            for child in token.children:
                edges.append(((token.text,token,token.dep_,token.i),
                              (child.text,child,child.dep_,child.i)))
    graph = nx.Graph(edges)


    path = []
    

    src = getSrc(sourceSpan)
    trg = getTrg(targetSpan)

    if(src == None or trg == None):
        return None, None


    source = (src.text, src, src.dep_,src.i)
    target = (trg.text, trg, trg.dep_,trg.i)
    try:
        shortestPath = nx.shortest_path(graph, source=source, target=target)
    except nx.exception.NodeNotFound:
        print("docname",docName)
        print("source span",sourceSpan)
        print("target span",targetSpan)
        print(f"Either source {source} or target {target} is not in G")
        print(sentences)
        return None, None



    if(len(shortestPath) < 2):
        print("error path shorter than 2")
        print("Source:", source, "target:",target)
        print("shortestpath:",shortestPath)
        print("docName:",docName)

    return shortestPath, {"source":src,"target":trg}



def getPathFreq(cont,source="Quantity",target="MeasuredEntity"):

    def getSource(srcSpan):
        temp = []
        for tok in srcSpan:
            if tok.tag_ == "CD":
                temp.append(tok)

        if temp == []:
            #print("No CD found in",[(x,x.pos_) for x in srcSpan])
            for tok in srcSpan:
                #print("$$Source replaced with:",(tok,tok.pos_), "span:",[(x,x.pos_) for x in srcSpan])
                return tok
        else:
            return temp[-1]

    def getTargetME(trgSpan):
        
        for x in reversed(trgSpan):
            if(x.pos_ in ["NOUN","PROPN"]):
                return x

        if(len(trgSpan) == 1):
            for tok in trgSpan:
                print("@@Target replaced with:",(tok,tok.pos_), "span:",[(x,x.pos_) for x in trgSpan])
                return tok
        #print("No noun found in",[(x,x.pos_) for x in trgSpan])

        return None

    def getTarget(trgSpan):
        
        for x in reversed(trgSpan):
            if(x.pos_ in ["NOUN","PROPN"]):
                return x

        if(len(trgSpan) == 1):
            for tok in trgSpan:
                #print("@@Target replaced with:",(tok,tok.pos_), "span:",[(x,x.pos_) for x in trgSpan])
                return tok
        #print("No noun found in",[(x,x.pos_) for x in trgSpan])

        return None
    

    accumPaths = []
    for e in cont.data.values(): 
        for x in e.doc._.meAnnots.values():
            try:
                path, info = getGraphPaths( x["sentences"], x[source], x[target],getSource, getTarget, reverse=False, pos=False, docName=e.name)
                if path == None or info == None:
                    continue

                accumPaths.append({
                    "path":path,
                    "doc":e.name,
                    "source":info["source"],
                    "target":info["target"]
                })
            except KeyError:
                pass
        

    def getPath(l1):
        #print(l1)
        x=0
        path = []
        while(x<len(l1)-1):
            
            if(l1[x][1].head.text == l1[x+1][1].text):
                #if x+1 is governor of x then
                path.append(l1[x][1].dep_)
            elif(l1[x][1].text == l1[x+1][1].head.text):
                path.append("r"+l1[x+1][1].dep_)
            else:
                print("error")
            
            x+=1
        #print(",".join(path))
        return tuple(path),l1[0][1].sent

    d={}
    for path in accumPaths:
        if path != None and path != []:
            count =0
            tempPath, tempSent = getPath(path["path"])
            del path["path"]  
            path["sent"] = tempSent

            try:
                d[tempPath][0] += 1
                d[tempPath][1].append(path)
            except KeyError: 
                d[tempPath] = [1,[path]]

    return {k: v for k, v in sorted(d.items(), key=lambda item : item[1][0],reverse=True)}   


    

