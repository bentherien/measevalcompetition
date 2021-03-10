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
        nodeIds = {}

        alldocs = list(self.data.values())
        random.shuffle(alldocs)
        for e in tqdm(alldocs):
            count+=1
            testjson  = e.doc.to_json()
            prevEnd = -1
            
            for sent in e.doc.sents:
                for token in sent:
                    prevEnd = createNode(token,e.doc,offset,xmlFile,prevEnd)\

                    start = offset + e.doc[token.i:token.i+1].start_char
                    try:
                        nodeIds[start]+=1
                    except KeyError:
                        nodeIds[start]=1
                    
                    
                    end = offset + e.doc[token.i:token.i+1].end_char
                    try:
                        nodeIds[end]+=1
                    except KeyError:
                        nodeIds[end]=1


            
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
                
                    
            # for index, row in e.tsv.iterrows():
            #     tempAnnot = {}
            #     tempAnnot["annotSet"] = row["annotSet"]
            #     tempAnnot["annotId"] = row["annotId"]
            #     tempAnnot["text"] = row["text"]
            #     if(type(row["other"]) == str):
            #         tempAnnot["other"] = row["other"]
            #     else:
            #         tempAnnot["other"] = "nothing"
                    
            #     annotz.append([annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile])    
            #     #createAnnotation(annotId, "MEval-"+row["annotType"] , offset + row["startOffset"], offset + row["endOffset"], tempAnnot, xmlFile)
            #     annotId += 1

            for x in e.doc._.meAnnotTrain:
                tempAnnot = {}
                try: 
                    tok = x["Quantity"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)

                    annotz.append([annotId, "MEval-Quantity", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["MeasuredEntity"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-MeasuredEntity", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["MeasuredProperty"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-MeasuredProperty", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass

                tempAnnot = {}
                try: 
                    tok = x["Qualifier"]
                    tempAnnot["text"] = tok.text
                    tempAnnot["spacy-pos"] = []
                    tempAnnot["spacy-dep"] = []
                    tempAnnot["spacy-head"] = []
                    tempAnnot["spacy-headhead"] = []
                    tempAnnot["spacy-headheadhead"] = []
                    for y in tok:
                        tempAnnot["spacy-pos"].append(y.tag_)
                        tempAnnot["spacy-dep"].append(y.dep_)
                        tempAnnot["spacy-head"].append(y.head.text)
                        tempAnnot["spacy-headhead"].append(y.head.head.text)
                        tempAnnot["spacy-headheadhead"].append(y.head.head.head.text)
                    annotz.append([annotId, "MEval-Qualifier", offset + tok.start_char, offset + tok.end_char, tempAnnot, xmlFile])
                    annotId += 1
                except KeyError:
                    pass
            
                    
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
            try:
                nodeIds[x[2]]+=1
                nodeIds[x[3]]+=1
                if(x[2]>=x[3]):
                    print("weird span: ({},{}) for {}".format(x[2],x[3],x))
                else:
                    createAnnotation(*x)
            except KeyError:
                print("unmatched NodeId for {}".format(x))
            
        xmlFile.write("</AnnotationSet>")
        xmlFile.write("</GateDocument>")    

        xmlFile.close()
        txtFile.close()



def getDataPos(self, fold, syspath, div = 8):
        test, train = self.getFolds(fold, div)

        with open(os.path.join(syspath,"train.txt"), "w", encoding="utf-8") as f:
            for x in train:
                f.write(x+"\n")

        with open(os.path.join(syspath,"test.txt"), "w", encoding="utf-8") as f:
            for x in test:
                f.write(x+"\n")

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")

        train_allS = open(os.path.join(datapath, "train_allS.tsv"),"w",encoding="utf-8")
        train_quantS = open(os.path.join(datapath, "train_quantS.tsv"),"w",encoding="utf-8")
        train_qualS = open(os.path.join(datapath, "train_qualS.tsv"),"w",encoding="utf-8")
        train_meS = open(os.path.join(datapath, "train_meS.tsv"),"w",encoding="utf-8")
        train_mpS = open(os.path.join(datapath, "train_mpS.tsv"),"w",encoding="utf-8")

        for i,x in enumerate(train):
            for sent in self.data[x].doc.sents:
                for token in sent:
                    train_allS.write(token.text+"\t"+token._.all+"\t"+token.tag_+"\n")
                    train_quantS.write(token.text+"\t"+token._.quant+"\t"+token.tag_+"\n")
                    train_qualS.write(token.text+"\t"+token._.qual+"\t"+token.tag_+"\n")
                    train_meS.write(token.text+"\t"+token._.me+"\t"+token.tag_+"\n")
                    train_mpS.write(token.text+"\t"+token._.mp+"\t"+token.tag_+"\n")


                train_allS.write("\n")
                train_quantS.write("\n")
                train_qualS.write("\n")
                train_meS.write("\n")
                train_mpS.write("\n")

        train_allS.close()
        train_quantS.close()
        train_qualS.close()
        train_meS.close()
        train_mpS.close()

        test_allS = open(os.path.join(datapath, "test_allS.tsv"),"w",encoding="utf-8")
        test_quantS = open(os.path.join(datapath, "test_quantS.tsv"),"w",encoding="utf-8")
        test_qualS = open(os.path.join(datapath, "test_qualS.tsv"),"w",encoding="utf-8")
        test_meS = open(os.path.join(datapath, "test_meS.tsv"),"w",encoding="utf-8")
        test_mpS = open(os.path.join(datapath, "test_mpS.tsv"),"w",encoding="utf-8")

        for i,x in enumerate(test):
            for sent in self.data[x].doc.sents:
                for token in sent:
                    test_allS.write(token.text+"\t"+token._.all+"\t"+token.tag_+"\n")
                    test_quantS.write(token.text+"\t"+token._.quant+"\t"+token.tag_+"\n")
                    test_qualS.write(token.text+"\t"+token._.qual+"\t"+token.tag_+"\n")
                    test_meS.write(token.text+"\t"+token._.me+"\t"+token.tag_+"\n")
                    test_mpS.write(token.text+"\t"+token._.mp+"\t"+token.tag_+"\n")

                
                test_allS.write("\n")
                test_quantS.write("\n")
                test_qualS.write("\n")
                test_meS.write("\n")
                test_mpS.write("\n")

        test_allS.close()
        test_quantS.close()
        test_qualS.close()
        test_meS.close()
        test_mpS.close()


def getDataPosDepSent(self, fold, syspath, div = 8):
        test, train = self.getFolds(fold, div)

        with open(os.path.join(syspath,"train.txt"), "w", encoding="utf-8") as f:
            for x in train:
                f.write(x+"\n")

        with open(os.path.join(syspath,"test.txt"), "w", encoding="utf-8") as f:
            for x in test:
                f.write(x+"\n")

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")

        train_allS = open(os.path.join(datapath, "train_allS.tsv"),"w",encoding="utf-8")
        train_quantS = open(os.path.join(datapath, "train_quantS.tsv"),"w",encoding="utf-8")
        train_qualS = open(os.path.join(datapath, "train_qualS.tsv"),"w",encoding="utf-8")
        train_meS = open(os.path.join(datapath, "train_meS.tsv"),"w",encoding="utf-8")
        train_mpS = open(os.path.join(datapath, "train_mpS.tsv"),"w",encoding="utf-8")
        
        for i,x in enumerate(train):
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    train_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    train_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                

                train_allS.write("\n")
                train_quantS.write("\n")
                train_qualS.write("\n")
                train_meS.write("\n")
                train_mpS.write("\n")

        train_allS.close()
        train_quantS.close()
        train_qualS.close()
        train_meS.close()
        train_mpS.close()

        test_allS = open(os.path.join(datapath, "test_allS.tsv"),"w",encoding="utf-8")
        test_quantS = open(os.path.join(datapath, "test_quantS.tsv"),"w",encoding="utf-8")
        test_qualS = open(os.path.join(datapath, "test_qualS.tsv"),"w",encoding="utf-8")
        test_meS = open(os.path.join(datapath, "test_meS.tsv"),"w",encoding="utf-8")
        test_mpS = open(os.path.join(datapath, "test_mpS.tsv"),"w",encoding="utf-8")

        for i,x in enumerate(test):
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    test_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    test_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")

                
                test_allS.write("\n")
                test_quantS.write("\n")
                test_qualS.write("\n")
                test_meS.write("\n")
                test_mpS.write("\n")

        test_allS.close()
        test_quantS.close()
        test_qualS.close()
        test_meS.close()
        test_mpS.close()


    def getDataPosDepDoc(self, fold, syspath, div = 8):
        test, train = self.getFolds(fold, div)

        with open(os.path.join(syspath,"data-fold{}".format(fold),"train.txt"), "w", encoding="utf-8") as f:
            for x in train:
                f.write(x+"\n")

        with open(os.path.join(syspath,"data-fold{}".format(fold),"test.txt"), "w", encoding="utf-8") as f:
            for x in test:
                f.write(x+"\n")

        datapath = os.path.join(syspath,"data-fold{}".format(fold))

        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        else:
            starpath = os.path.join(datapath,"*") 
            os.system(f"rm {starpath}")

        train_allS = open(os.path.join(datapath, "train_allS.tsv"),"w",encoding="utf-8")
        train_quantS = open(os.path.join(datapath, "train_quantS.tsv"),"w",encoding="utf-8")
        train_qualS = open(os.path.join(datapath, "train_qualS.tsv"),"w",encoding="utf-8")
        train_meS = open(os.path.join(datapath, "train_meS.tsv"),"w",encoding="utf-8")
        train_mpS = open(os.path.join(datapath, "train_mpS.tsv"),"w",encoding="utf-8")
        
        for i,x in enumerate(train):
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    train_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    train_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                

                train_allS.write("\n")
                train_quantS.write("\n")
                train_qualS.write("\n")
                train_meS.write("\n")
                train_mpS.write("\n")

        train_allS.close()
        train_quantS.close()
        train_qualS.close()
        train_meS.close()
        train_mpS.close()

        test_allS = open(os.path.join(datapath, "test_allS.tsv"),"w",encoding="utf-8")
        test_quantS = open(os.path.join(datapath, "test_quantS.tsv"),"w",encoding="utf-8")
        test_qualS = open(os.path.join(datapath, "test_qualS.tsv"),"w",encoding="utf-8")
        test_meS = open(os.path.join(datapath, "test_meS.tsv"),"w",encoding="utf-8")
        test_mpS = open(os.path.join(datapath, "test_mpS.tsv"),"w",encoding="utf-8")

        for i,x in enumerate(test):
            count=0
            sOffset=0
            for sent in self.data[x].doc.sents:
                sOffset += count
                count = 0
                for token in sent:
                    count += 1
                    test_allS.write("\t".join([token.text,token._.all,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")
                    test_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i),token.dep_,str(token.head.i)]+[str(x) for x in token._.relationDoc])+"\n")

                
                test_allS.write("\n")
                test_quantS.write("\n")
                test_qualS.write("\n")
                test_meS.write("\n")
                test_mpS.write("\n")

        test_allS.close()
        test_quantS.close()
        test_qualS.close()
        test_meS.close()
        test_mpS.close()



    def getEncodingSentNihatha(self, fold, syspath, skip=[], div = 8):
            self.annotateRelations2()

            test, train = self.getFolds(fold, div)
            print(os.path.join(syspath,"data-fold{}-last".format(fold), "train.txt"))

            datapath = os.path.join(syspath,"data-fold{}-last".format(fold))

            if not os.path.isdir(datapath):
                os.mkdir(datapath)

            with open(os.path.join(syspath,"data-fold{}-last".format(fold), "train.txt"), "w", encoding="utf-8") as f:
                for x in train:
                    f.write(x+"\n")
                f.close()

            with open(os.path.join(syspath,"data-fold{}-last".format(fold), "test.txt"), "w", encoding="utf-8") as f:
                for x in test:
                    f.write(x+"\n")
                f.close()

            

            train_allS = open(os.path.join(datapath, "train_allS.tsv"),"w",encoding="utf-8")
            train_quantS = open(os.path.join(datapath, "train_quantS.tsv"),"w",encoding="utf-8")
            train_qualS = open(os.path.join(datapath, "train_qualS.tsv"),"w",encoding="utf-8")
            train_meS = open(os.path.join(datapath, "train_meS.tsv"),"w",encoding="utf-8")
            train_mpS = open(os.path.join(datapath, "train_mpS.tsv"),"w",encoding="utf-8")

            allcount = 0
            for i,y in enumerate(train):
                if y in skip: 
                    continue
                count=0
                sOffset=0
                sentCount = -1
                for sent in self.data[y].doc.sents:
                    sentCount += 1
                    sOffset += count
                    count = 0
                    for token in sent:
                        count += 1
                        allcount +=1
                        train_allS.write("\t".join([token.text,token._.all.upper() if token._.all != "QA" else "Q",token.tag_,str(token._.newid),token.dep_,str(token.head._.newid)]+[str(x) for x in range(4)])+"\n")
                        train_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        train_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        train_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        train_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    
                    train_allS.write("@@Annotations@@\n")
                    temp = ["QA"]
                    for i,x in enumerate(sent._.QA2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["ME"]
                    for i,x in enumerate(sent._.ME2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["MP"]
                    for i,x in enumerate(sent._.MP2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["QL"]
                    for i,x in enumerate(sent._.QL2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_ME_Rel"]
                    for i,x in enumerate(sent._.qa_me_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_MP_Rel"]
                    for i,x in enumerate(sent._.qa_mp_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["MP_ME_Rel"]
                    for i,x in enumerate(sent._.mp_me_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_QL_Rel"]
                    for i,x in enumerate(sent._.qa_ql_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    train_allS.write("\t".join(temp)+"\n")
                    # temp = ["modifiers"]
                    # for i,x in enumerate(sent._.modifiers):
                    #     for mod in x:
                    #         temp.append("{}:{}".format(i,mod))
                    # train_allS.write("\t".join(temp)+"\n")
                    train_allS.write("\t".join(["DocID",str(y)])+"\n")
                    train_allS.write("\t".join(["SentID",str(sentCount)])+"\n")

                    train_allS.write("\n")
                    train_quantS.write("\n")
                    train_qualS.write("\n")
                    train_meS.write("\n")
                    train_mpS.write("\n")

            train_allS.close()
            train_quantS.close()
            train_qualS.close()
            train_meS.close()
            train_mpS.close()

            test_allS = open(os.path.join(datapath, "test_allS.tsv"),"w",encoding="utf-8")
            test_quantS = open(os.path.join(datapath, "test_quantS.tsv"),"w",encoding="utf-8")
            test_qualS = open(os.path.join(datapath, "test_qualS.tsv"),"w",encoding="utf-8")
            test_meS = open(os.path.join(datapath, "test_meS.tsv"),"w",encoding="utf-8")
            test_mpS = open(os.path.join(datapath, "test_mpS.tsv"),"w",encoding="utf-8")

            

            for i,y in enumerate(test):
                if y in skip: 
                    continue
                count=0
                sOffset=0
                sentCount=-1
                for sent in self.data[y].doc.sents:
                    sentCount +=1
                    sOffset += count
                    count = 0
                    for token in sent:
                        count += 1
                        test_allS.write("\t".join([token.text,token._.all.upper() if token._.all != "QA" else "Q",token.tag_,str(token._.newid),token.dep_,str(token.head._.newid)]+[str(x) for x in range(4)])+"\n")
                        test_quantS.write("\t".join([token.text,token._.quant,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        test_qualS.write("\t".join([token.text,token._.qual,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        test_meS.write("\t".join([token.text,token._.me,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                        test_mpS.write("\t".join([token.text,token._.mp,token.tag_,str(token.i-sOffset),token.dep_,str(token.head.i-sOffset)]+[str(x) for x in token._.relationSent])+"\n")
                    
                    test_allS.write("@@Annotations@@\n")
                    temp = ["QA"]
                    for i,x in enumerate(sent._.QA2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["ME"]
                    for i,x in enumerate(sent._.ME2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["MP"]
                    for i,x in enumerate(sent._.MP2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["QL"]
                    for i,x in enumerate(sent._.QL2):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_ME_Rel"]
                    for i,x in enumerate(sent._.qa_me_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_MP_Rel"]
                    for i,x in enumerate(sent._.qa_mp_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["MP_ME_Rel"]
                    for i,x in enumerate(sent._.mp_me_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    temp = ["QA_QL_Rel"]
                    for i,x in enumerate(sent._.qa_ql_rel):
                        temp.append("{}:({},{})".format(i,x[0],x[1]))
                    test_allS.write("\t".join(temp)+"\n")
                    # temp = ["modifiers"]
                    # for i,x in enumerate(sent._.modifiers):
                    #     for mod in x:
                    #         temp.append("{}:{}".format(i,mod))
                    # test_allS.write("\t".join(temp)+"\n")
                    test_allS.write("\t".join(["DocID",str(y)])+"\n")
                    test_allS.write("\t".join(["SentID",str(sentCount)])+"\n")
                    
                    test_allS.write("\n")
                    test_quantS.write("\n")
                    test_qualS.write("\n")
                    test_meS.write("\n")
                    test_mpS.write("\n")

            test_allS.close()
            test_quantS.close()
            test_qualS.close()
            test_meS.close()
            test_mpS.close()

    def annotateRelations2(self):
        # Token.set_extension("relationDoc", default=(0,"root",0), force=True)
        # Token.set_extension("relationSent", default=(0,"root",0), force=True)
        # Doc.set_extension("ME", default=[], force=True)
        # Doc.set_extension("MP", default=[], force=True)
        # Doc.set_extension("QL", default=[], force=True)
        # Doc.set_extension("QA", default=[], force=True)
        # Doc.set_extension("qa_me_rel", default=[], force=True)
        # Doc.set_extension("qa_mp_rel", default=[], force=True)
        # Doc.set_extension("mp_me_rel", default=[], force=True)
        # Doc.set_extension("qa_ql_rel", default=[], force=True)
        # Doc.set_extension("modifiers", default=[], force=True)

        Span.set_extension("ME2", default=[], force=True)
        Span.set_extension("MP2", default=[], force=True)
        Span.set_extension("QL2", default=[], force=True)
        Span.set_extension("QA2", default=[], force=True)
        Span.set_extension("qa_me_rel2", default=[], force=True)
        Span.set_extension("qa_mp_rel2", default=[], force=True)
        Span.set_extension("mp_me_rel2", default=[], force=True)
        Span.set_extension("qa_ql_rel2", default=[], force=True)
        Span.set_extension("modifiers2", default=[], force=True)

        Token.set_extension("newid", default=-1, force=True)

        count = 0
        for x in self.data.values():
            for token in x.doc:
                token._.newid = count
                count+=1


        tempDict = {"MeasuredEntity":"ME","MeasuredProperty":"MP","Quantity":"QA","Qualifier":"QL"}
        for x in self.data.values():
            e = x
            for annot in x.doc._.meAnnots.values():

                #at the document level
                spanQuant = annot["Quantity"]["span"].start
                e.doc._.QA.append((annot["Quantity"]["span"].start,annot["Quantity"]["span"].end- 1,))
                try:
                    e.doc._.modifiers.append(annot["Quantity"]["other"]["mods"])
                except KeyError:
                    e.doc._.modifiers.append(["Nomod"])
                if "MeasuredProperty" in annot:
                    propID = -1 
                    e.doc._.MP.append((annot["MeasuredProperty"]["span"].start,annot["MeasuredProperty"]["span"].end- 1,))
                    e.doc._.qa_mp_rel.append((len(e.doc._.QA)-1,len(e.doc._.MP)-1))
                    
                    for y in annot["MeasuredProperty"]["span"]:
                        propID = y.i
                        y._.relationDoc = (y.i, "HasQuantity", spanQuant)
                        break

                    if "MeasuredEntity" in annot:
                        e.doc._.ME.append((annot["MeasuredEntity"]["span"].start,annot["MeasuredEntity"]["span"].end- 1,))
                        e.doc._.mp_me_rel.append((len(e.doc._.MP)-1,len(e.doc._.ME)-1))
                        if propID != -1:
                            for y in annot["MeasuredEntity"]["span"]:
                                y._.relationDoc = (y.i, "HasPropety", propID)
                                break
                    else:
                        pass
                        #print("error no mesured entity, but property",e.name, annot)

                elif "MeasuredEntity" in annot:
                    e.doc._.ME.append((annot["MeasuredEntity"]["span"].start,annot["MeasuredEntity"]["span"].end- 1,))
                    e.doc._.qa_me_rel.append((len(e.doc._.QA)-1,len(e.doc._.ME)-1))
                    for y in annot["MeasuredEntity"]["span"]:
                        y._.relationDoc = (y.i, "HasQuantity", spanQuant)
                        break

                try: 
                    for y in annot["Qualifier"]["span"]:
                        y._.relationDoc = (y.i, "Qualifies", spanQuant)
                        e.doc._.QL.append((annot["Qualifier"]["span"].start,annot["Qualifier"]["span"].end- 1,))
                        e.doc._.qa_ql_rel.append((len(e.doc._.QA)-1,len(e.doc._.QL)-1))
                        break
                except KeyError:
                    pass
                

                #at the sentence level
                if len(annot["sentences"]) == 1:
                    sent = annot["sentences"][0]
                    ss = annot["sentences"][0].start
                    spanQuant = annot["Quantity"]["span"].start
                    ts=e.doc[annot["Quantity"]["span"].start]._.newid
                    te=e.doc[annot["Quantity"]["span"].end - 1]._.newid
                    sent._.QA2.append((ts,te,))
                    try:
                        sent._.modifiers.append(annot["Quantity"]["other"]["mods"])
                    except KeyError:
                        sent._.modifiers.append(["Nomod"])

                    if "MeasuredProperty" in annot:
                        propID = -1
                        ts=e.doc[annot["MeasuredProperty"]["span"].start]._.newid
                        te=e.doc[annot["MeasuredProperty"]["span"].end - 1]._.newid
                        sent._.MP2.append((ts,te,))
                        sent._.qa_mp_rel.append((len(sent._.QA)-1,len(sent._.MP)-1))
                        for y in annot["MeasuredProperty"]["span"]:
                            propID = y.i
                            y._.relationSent = (y.i - ss, "HasQuantity", spanQuant - ss)
                            break
                        if "MeasuredEntity" in annot:
                            ts=e.doc[annot["MeasuredEntity"]["span"].start]._.newid
                            te=e.doc[annot["MeasuredEntity"]["span"].end - 1]._.newid
                            sent._.ME2.append((ts,te,))
                            sent._.mp_me_rel.append((len(sent._.MP)-1,len(sent._.ME)-1))
                            if propID != -1:
                                for y in annot["MeasuredEntity"]["span"]:
                                    y._.relationSent = (y.i - ss, "HasPropety", propID - ss)
                                    break
                        else:
                            pass
                            #print("error no mesured entity, but property",e.name, annot)
                    elif "MeasuredEntity" in annot:
                        ts=e.doc[annot["MeasuredEntity"]["span"].start]._.newid
                        te=e.doc[annot["MeasuredEntity"]["span"].end - 1]._.newid
                        sent._.ME2.append((ts,te,))
                        sent._.qa_me_rel.append((len(sent._.QA)-1,len(sent._.ME)-1))
                        for y in annot["MeasuredEntity"]["span"]:
                            y._.relationSent = (y.i - ss, "HasQuantity", spanQuant - ss)
                            break

                    try: 
                        for y in annot["Qualifier"]["span"]:
                            ts=e.doc[annot["Qualifier"]["span"].start]._.newid
                            te=e.doc[annot["Qualifier"]["span"].end - 1]._.newid
                            sent._.QL2.append((ts,te,))
                            sent._.qa_ql_rel.append((len(sent._.QA)-1,len(sent._.QL)-1))
                            y._.relationSent = (y.i - ss, "Qualifies", spanQuant - ss)
                            break
                    except KeyError:
                        pass
                else:
                    doc = e.doc
                    sent = doc[annot["Quantity"]["span"].start].sent
                    ss = sent.start
                    ts=e.doc[annot["Quantity"]["span"].start]._.newid
                    te=e.doc[annot["Quantity"]["span"].end - 1]._.newid
                    sent._.QA2.append((ts,te,))
                    try:
                        sent._.modifiers.append(annot["Quantity"]["other"]["mods"])
                    except KeyError:
                        sent._.modifiers.append(["Nomod"])
                    if "MeasuredProperty" in annot:
                        sent = doc[annot["MeasuredProperty"]["span"].start].sent
                        ss = sent.start
                        ts=e.doc[annot["MeasuredProperty"]["span"].start]._.newid
                        te=e.doc[annot["MeasuredProperty"]["span"].end - 1]._.newid
                        sent._.MP2.append((ts,te,))

                        if "MeasuredEntity" in annot:
                            sent = doc[annot["MeasuredEntity"]["span"].start].sent
                            ss = sent.start
                            ts=e.doc[annot["MeasuredEntity"]["span"].start]._.newid
                            te=e.doc[annot["MeasuredEntity"]["span"].end - 1]._.newid
                            sent._.ME2.append((ts,te,))
                        else:
                            pass
                            #print("error no mesured entity, but property",e.name, annot)

                    elif "MeasuredEntity" in annot:
                        sent = doc[annot["MeasuredEntity"]["span"].start].sent
                        ss = sent.start
                        ts=e.doc[annot["MeasuredEntity"]["span"].start]._.newid
                        te=e.doc[annot["MeasuredEntity"]["span"].end - 1]._.newid
                        sent._.ME2.append((ts,te,))

                    if "Qualifier" in annot:
                        sent = doc[annot["Qualifier"]["span"].start].sent
                        ss = sent.start
                        ts=e.doc[annot["Qualifier"]["span"].start]._.newid
                        te=e.doc[annot["Qualifier"]["span"].end - 1]._.newid
                        sent._.QL2.append((ts,te,))
