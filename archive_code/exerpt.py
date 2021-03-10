
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


