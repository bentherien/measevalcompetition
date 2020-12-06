import common

def hline():
    return "\\hline\n"

def tableStart(label="def label", caption="def caption", arrangement="x|x|X", size="300pt"):
    a = "\\begin{table}[!ht] \n"
    b = "\\centering\n"
    c = "\\caption{{{}}}\\label{{{}}}\n".format(caption,label)
    d = "\\begin{{tabularx}}{{{}}}{{{}}}\n".format(size, arrangement)
    return a + b + c + d + hline()

def tableEnd():
    return "\\end{tabularx}\n\\end{table}"

def writeCReport(report, fileObj, label="def label", caption="def caption", arrangement="|X|X|X|X|X|", size="300pt",epoch=0, fold=1):
    temp = [ x.split(" ") for x in report.split("\n") if (x.split(" ")!=[""])]
    temp = [[y for y in z if y!=""] for z in temp]
    fileObj.write(tableStart(label,caption,arrangement,size))
    width = len([x for x in arrangement if x == "|"])-1
    for x in temp:
        if x[0] in ["QL", "ME", "MP", "QA"]:
            y = x[:]
            y.insert(0,str(epoch))
            y.insert(0,str(fold))
            common.f.write("&".join(y)+"\\\\\n"+hline())
            common.f.flush()

        
        while len(x) < width:
            x.insert(0,"")
        if len(x) == 6:
            ttemp = x[0]+" "+x[1]
            x.pop(0)
            x.pop(0)
            x.insert(0,ttemp)
             
        fileObj.write("&".join(x)+"\\\\\n")
        fileObj.write(hline())

    fileObj.write(tableEnd())


def writeSummary(fileObj, label="def label", caption="def caption", arrangement="|X|X|X|X|X|X|X|", size="300pt"):
    fileObj.write(tableStart(label,caption,arrangement,size))
    fileObj.write("fold&epoch&&precision&recall&f1-score&support\n")
    fileObj.write(hline())
    width = len([x for x in arrangement if x == "|"])-1
    for x in common.summary:     
        fileObj.write("&".join(x)+"\\\\\n")
        fileObj.write(hline())

    fileObj.write(tableEnd())




def printFoldInfo():

    def getSizeData(data):
        last = ""
        freq = {}
        for x in data:
            for y in x.target:
                if y == last: 
                    continue
                else:
                    last = y
                    try:
                        freq[y] += 1
                    except KeyError:
                        freq[y] = 1
        del freq["o"]
        return freq
    print("&".join(["fold","type","QA","ME","MP","QL"])+"\\\\\n\\hline")
    for fold in range(1,9):
        testData, train = getFolds(fold, all_data, 8)
        testSize, trainSize = getSizeData(testData), getSizeData(train)
        
        ltemp = [str(fold), "test"]
        for k in ["ME","MP","QL","QA"]:
            ltemp.append(str(testSize[k]))
        print("&".join(ltemp)+"\\\\\n\\hline")
        ltemp = [str(fold), "train"]
        for k in ["ME","MP","QL","QA"]:
            ltemp.append(str(trainSize[k]))
        print("&".join(ltemp)+"\\\\\n\\hline")
    exit(0)


def getSummary(epoch,fold,embeddingType,learningRate,report,title):
    temp = [ x.split(" ") for x in report.split("\n") if (x.split(" ")!=[""])]
    temp = [[y for y in z if y!=""] for z in temp]
    l = []
    for x in temp:
        if x[0].upper() in ["QL", "ME", "MP", "QA"]:
            l.append({
                "title":title,
                "epoch":epoch,
                "annot":x[0],
                "F1":x[3],
                "precision":x[1],
                "recall":x[2],
                "support":x[4],
                "lr":learningRate,
                "fold":fold,
                "emb":embeddingType
                })
    return l

