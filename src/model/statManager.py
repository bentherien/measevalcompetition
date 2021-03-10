

class StatManger:

    def processReport(self,report):
        temp = [x.split(" ") for x in report.split("\n") if (x.split(" ")!=[""])]
        temp = [[y for y in z if y!=""] for z in temp]
        s = ""
        for x in temp:
            if x[0] in ["QL", "ME", "MP", "QA",'o']:
                s += "&".join(x)+"\\\\\n"
            elif x[0] == "accuracy":
                y = x[:]
                y.insert(1,"")
                y.insert(1,"")
                s += "&".join(y)+"\\\\\n"
            elif x[0] == "weighted":
                y = x[:]
                y.pop(0)
                y[0] = "weighted avg"
                s += "&".join(y)+"\\\\\n"
            elif x[0] == "macro":
                y = x[:]
                y.pop(0)
                y[0] = "macro avg"
                s += "&".join(y)+"\\\\\n"
            elif x[0] == "precision":
                y = x[:]
                y.insert(0,"category")
                s += "&".join(y)+"\\\\\n"
        return s


    def __init__(self,report):
            