from code.helpers import intersectSpan
import pprint

def getAscii(e, fileObj):
    text = ""
    count = 0
    counts = []
    dobreak = False
    for sent in e.doc.sents: 
        count+=1
        if dobreak == False and count > 1:
            pass

        for meas in e.measurements.values():
            dobreak=False
            for annot in meas.values():
                if(intersectSpan(sent,annot["startOffset"],annot["endOffset"])):
                    counts.append(count)
                    text += sent.text
                    dobreak=True
                    break
                    
            if dobreak:
                break
            
        
    fileObj.write(f"Document {e.name}:\n\n")
    fileObj.write(text+"\n\n")
    
    fileObj.write("Gold Annotations:\n\n")
    fp = open(f"data-merged/tsv/{e.name}.tsv","r",encoding="utf-8")
    fileObj.write(fp.read())
    fp.close()
    
    fileObj.write("\n\nHypothesis 0 annotations:\n")
    count =0
    
    for meas in e.doc._.h0Measurements:
        count+=1
        fileObj.write(f"\nMeasurement {count}\n")
        num = meas["Number"]
        unit = meas["Unit"]
        me = meas["MeasuredEntity"]
        fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
        fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
        fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))

        
    fileObj.write("\n\nCorrect Hypothesis 0 annotations:\n")
    if len(e.doc._.h0MeasurementTps) > 0:
        

        for meas in e.doc._.h0MeasurementTps:
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
    for sentence in e.doc.sents:
        count+=1
        if count in counts:
            print(f"Sentence {count}: \n\n", sentence, "\n",file = fileObj)
            print("governor".ljust(15), "dependency".ljust(15), "token".ljust(15),file = fileObj)
            fileObj.write("-----------------------------------------\n")
            for token in sentence:
                print(token.head.text.ljust(15),token.dep_.ljust(15), token.text.ljust(15),file = fileObj)
            print("\n\n",file = fileObj)



def getAsciiQ(e, fileObj, constituency = False):
    text = ""
    count = 0
    counts = []
    dobreak = False
    for sent in e.doc.sents: 
        count+=1
        if dobreak == False and count > 1:
            pass
        
        for meas in e.measurements.values():
            dobreak=False
            for annot in meas.values():
                if(intersectSpan(sent,annot["startOffset"],annot["endOffset"])):
                    counts.append(count)
                    text += sent.text
                    dobreak=True
                    break
                    
            if dobreak:
                break
            
        
    fileObj.write(f"Document {e.name}:\n\n")
    fileObj.write(text+"\n\n")
    
    fileObj.write("Gold Annotations:\n\n")
    fp = open(f"data-merged/tsv/{e.name}.tsv","r",encoding="utf-8")
    fileObj.write(fp.read())
    fp.close()

    print("\n\nGovernor".ljust(15), "Dependency".ljust(15), "Quantity".ljust(15),file = fileObj)
    fileObj.write("-----------------------------------------\n")
    for x in e.doc._.meAnnots.values():
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
    
    for meas in e.doc._.h0Measurements:
        count+=1
        fileObj.write(f"\nMeasurement {count}\n")
        num = meas["Number"]
        unit = meas["Unit"]
        me = meas["MeasuredEntity"]
        fileObj.write("Number          - startOffset:{}, endOffset:{}, text:{}\n".format(num["start"],num["end"],num["text"]))
        fileObj.write("unit            - startOffset:{}, endOffset:{}, text:{}\n".format(unit["start"],unit["end"],unit["text"]))
        fileObj.write("Measured Entity - startOffset:{}, endOffset:{}, text:{}\n".format(me["start"],me["end"],me["text"]))

        
    fileObj.write("\n\nCorrect Hypothesis 0 annotations:\n")
    if len(e.doc._.h0MeasurementTps) > 0:
        

        for meas in e.doc._.h0MeasurementTps:
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
    for sentence in e.doc.sents:
        count+=1
        if count in counts:
            print(f"Sentence {count}: \n\n", sentence, "\n",file = fileObj)
            if constituency:
                pp = pprint.PrettyPrinter(indent=3, width=80, depth=5,  stream = fileObj)
                print("\nConstituency parse: \n\n",file = fileObj)
                print(sentence._.parse_string,file = fileObj)
                #pp.pprint(sentence._.parse_string)
                print("\n\n",file = fileObj)
            print("governor".ljust(15), "dependency".ljust(15), "token".ljust(15),file = fileObj)
            fileObj.write("-----------------------------------------\n")
            for token in sentence:
                print(token.head.text.ljust(15),token.dep_.ljust(15), token.text.ljust(15),file = fileObj)
            print("\n\n",file = fileObj)