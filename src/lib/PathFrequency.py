import numpy as np
from src.lib.extra import getPathFreq
import os
import random

class PathFrequecy:

    def __init__(self, excontroller, source, target):
        self.target = target
        self.source = source 
        self.freq = getPathFreq(excontroller, source, target)

        
    def getTotal(self):
        return np.sum([x[0] for x in self.freq.values()])

    def getFileName(self,key):
        return "{}-freq{}.txt".format("-".join(list(key)),self.freq[key][0])

    def writeAscii(self,maxfile=10,maxsent=3):
        if( not os.path.isdir("asciifreq")):
            os.mkdir("asciifreq")

        if (os.path.isdir(os.path.join("asciifreq",f"{self.source}-{self.target}"))):
            os.system("rm {}".format(os.path.join("asciifreq",f"{self.source}-{self.target}","*")))
        else:
            os.mkdir(os.path.join("asciifreq",f"{self.source}-{self.target}"))

        outsideCount=0
        for key in self.freq.keys():
            if("appos" in key):
                fileObj = open(os.path.join("asciifreq",f"{self.source}-{self.target}",self.getFileName(key)),"w",encoding="utf-8")
                fileObj.write("Frequency: {}\n".format(self.freq[key][0]))
                fileObj.write("Path: {}\n".format(key))
                fileObj.write("Type: {}\n".format(f"{self.source}-{self.target}"))

                count = 0
                tempList = self.freq[key][1]
                random.shuffle(tempList)
                for info in tempList:
                    sentence = info["sent"]

                    fileObj.write("\n"+sentence.text+"\n")
                    fileObj.write("{}: {}\n".format(self.source,info["source"]))
                    fileObj.write("{}: {}\n".format(self.target,info["target"]))

                    fileObj.write("\n\nDependencies:\n")
                    count =0
                    print("governor".ljust(15), "dependency".ljust(15), "token".ljust(15),file = fileObj)
                    fileObj.write("-----------------------------------------\n")
                    for token in sentence:
                        print(token.head.text.ljust(15),token.dep_.ljust(15), token.text.ljust(15),file = fileObj)
                    print("\n\n",file = fileObj)

                    count+=1
                    if count == maxsent: 
                        break

                fileObj.close()

                outsideCount+=1
                if outsideCount == maxfile: 
                    break

    def __getitem__(self, key):
        return self.freq[key]
        


    def getLatex(self, maxCount=43):
        print(startTable(caption="Most Frequent Paths from {} to {}. Total paths found={}".format(self.source,self.target,self.getTotal()),label="{}-{}".format(self.source,self.target), elements="|X|x|", length = "350pt"))
        print("\hline\nPattern&frequency\\\\\n\\hline")
        g = {k: v[0] for k, v in sorted(self.freq.items(), key=lambda item: item[1][0],reverse=True)}
        count = 0
        for x in g.keys():
            count+=1
            if count == maxCount:
                break
                
            print(",".join(list(x)),"&",g[x],"\\\\\n\\hline")

        print(endTable())
        print("\\clearpage")

    def getTexAt(self, length, maxCount=43):
        temp = self.getFrequencyAtLength(length)
        templen = np.sum([v for k,v in temp.items()])
        print(startTable(caption="Frequency of starting patterns of length {} between {} and {}. Total paths counted={}".format(length,self.source,self.target,templen),label="{}-{}".format(self.source,self.target), elements="|X|x|", length = "350pt"))
        print("\hline\nPattern&frequency\\\\\n\\hline")
        count = 0
        for x in temp.keys():
            count+=1
            if count == maxCount:
                break
                
            print(",".join(list(x)),"&",temp[x],"\\\\\n\\hline")
        print(endTable())
        print("\\clearpage")

    def getTexAtPos(self, length, maxCount=43):
        temp = self.getFrequencyAtPosition(length)
        templen = np.sum([v for k,v in temp.items()])
        print(startTable(caption="Frequency of transition at position {} in paths between {} and {}. Total paths counted={}".format(length,self.source,self.target,templen),label="{}-{}".format(self.source,self.target), elements="|X|x|", length = "350pt"))
        print("\hline\nPattern&frequency\\\\\n\\hline")
        count = 0
        for x in temp.keys():
            count+=1
            if count == maxCount:
                break
                
            print(x,"&",temp[x],"\\\\\n\\hline")
        print(endTable())
        print("\\clearpage")

    def getFrequencyAtLength(self,length):
        paths = list(self.freq.keys())
        fal = {}

        for x in paths:
            if len(x) < length:
                pass
            else:
                try:
                    if type(x[:length]) == str:
                        fal[tuple([x[:length]])] += self.freq[x][0]
                    else:
                        fal[x[:length]] += self.freq[x][0]
                except KeyError:
                    if type(x[:length]) == str:
                        fal[tuple([x[:length]])] = self.freq[x][0]
                    else:
                        fal[x[:length]] = self.freq[x][0]
        return {k: v for k, v in sorted(fal.items(), key=lambda item: item[1],reverse=True)}

    def getFrequencyAtPosition(self,position):
        paths = list(self.freq.keys())
        fal = {}

        for x in paths:
            if len(x) - 1 < position:
                pass
            else:
                try:
                    fal[x[position]] += self.freq[x][0]
                except KeyError:
                    fal[x[position]] = self.freq[x][0]

        return {k: v for k, v in sorted(fal.items(), key=lambda item: item[1],reverse=True)}

from src.lib.texgen import *

class PFController:
    def __init__(self,a,b,c):
        self.me = a
        self.mp = b
        self.q = c
        self.getTable()
        
    def getTable(self):
        merged = {}

        for x in self.me.freq.keys():
            try:
                merged[x]["me"] += self.me.freq[x][0]
            except:
                merged[x] = {"me": self.me.freq[x][0], "mp": 0, "q":0}

        for x in self.mp.freq.keys():
            try:
                merged[x]["mp"] += self.mp.freq[x][0]
            except:
                merged[x] = {"me": 0, "mp": self.mp.freq[x][0], "q":0}

        for x in self.q.freq.keys():
            try:
                merged[x]["q"] += self.q.freq[x][0]
            except:
                merged[x] = {"me": 0, "mp": 0, "q":self.q.freq[x][0]}

        for x in merged.keys():
            merged[x]["total"] = merged[x]["me"] + merged[x]["mp"] + merged[x]["q"]

        self.merged = {k: v for k, v in sorted(merged.items(), key=lambda item: item[1]["total"],reverse=True)}


    def getFrequencyAtLength(self,length):
        tempME = self.me.getFrequencyAtLength(length)
        tempMP = self.mp.getFrequencyAtLength(length)
        tempQ = self.q.getFrequencyAtLength(length)

        merged = {}

        for x in tempME.keys():
            try:
                merged[x]["me"] += tempME[x]
            except:
                merged[x] = {"me": tempME[x], "mp": 0, "q":0}

        for x in tempMP.keys():
            try:
                merged[x]["mp"] += tempMP[x]
            except:
                merged[x] = {"me": 0, "mp": tempMP[x], "q":0}

        for x in tempQ.keys():
            try:
                merged[x]["q"] += tempQ[x]
            except:
                merged[x] = {"me": 0, "mp": 0, "q":tempQ[x]}

        for x in merged.keys():
            merged[x]["total"] = merged[x]["me"] + merged[x]["mp"] + merged[x]["q"]

        return {k: v for k, v in sorted(merged.items(), key=lambda item: item[1]["total"],reverse=True)}

    def getFrequencyAtPosition(self,position):
        tempME = self.me.getFrequencyAtPosition(position)
        tempMP = self.mp.getFrequencyAtPosition(position)
        tempQ = self.q.getFrequencyAtPosition(position)

        merged = {}

        for x in tempME.keys():
            try:
                merged[x]["me"] += tempME[x]
            except:
                merged[x] = {"me": tempME[x], "mp": 0, "q":0}

        for x in tempMP.keys():
            try:
                merged[x]["mp"] += tempMP[x]
            except:
                merged[x] = {"me": 0, "mp": tempMP[x], "q":0}

        for x in tempQ.keys():
            try:
                merged[x]["q"] += tempQ[x]
            except:
                merged[x] = {"me": 0, "mp": 0, "q":tempQ[x]}

        for x in merged.keys():
            merged[x]["total"] = merged[x]["me"] + merged[x]["mp"] + merged[x]["q"]

        return {k: v for k, v in sorted(merged.items(), key=lambda item: item[1]["total"],reverse=True)}

    def getTexAtPos(self, length, maxCount=43):
        temp = self.getFrequencyAtPosition(length)

        meCount = np.sum([v["me"] for k,v in temp.items()])
        mpCount = np.sum([v["mp"] for k,v in temp.items()])
        qCount = np.sum([v["q"] for k,v in temp.items()])
        total = meCount + mpCount + qCount

        print(startTable(caption="Frequency of transition at position {} in paths between quantity and measured entity({}), measured property({}), and qualifier({}). Total paths counted={}".format(length, meCount,mpCount,qCount,total),label="mp1", elements="|X|x|x|x|x|", length = "250pt"))
        print("\hline\nPattern&ME-freq&MP-freq&Q-freq&Total\\\\\n\\hline")
        count = 0
        for x in temp.keys():
            count+=1
            if count == maxCount:
                break
            print("{}&{}&{}&{}&{}\\\\\n\\hline".format(x,temp[x]["me"],temp[x]["mp"],temp[x]["q"],temp[x]["total"]))
        print(endTable())
        print("\\clearpage")

    def getTexAt(self,length, maxCount=42):
        temp = self.getFrequencyAtLength(length)

        meCount = np.sum([v["me"] for k,v in temp.items()])
        mpCount = np.sum([v["mp"] for k,v in temp.items()])
        qCount = np.sum([v["q"] for k,v in temp.items()])
        total = meCount + mpCount + qCount

        print(startTable(caption="Frequency of starting patterns of length {} from quantity to measured entity({}), measured property({}), and qualifier({}). Total paths counted={}".format(length, meCount,mpCount,qCount,total),label="mp1", elements="|X|x|x|x|x|", length = "400pt"))
        print("\hline\nPattern&ME-freq&MP-freq&Q-freq&Total\\\\\n\\hline")
        count = 0
        for x in temp.keys():
            count+=1
            if count == maxCount:
                break
            print("{}&{}&{}&{}&{}\\\\\n\\hline".format(",".join(list(x)),temp[x]["me"],temp[x]["mp"],temp[x]["q"],temp[x]["total"]))
        print(endTable())
        print("\\clearpage")

    def getLatex(self, maxCount=42):
        total = self.me.getTotal()+self.mp.getTotal()+self.q.getTotal()
        print(startTable(caption="Most Frequent Paths from quantity to measured entity({}), measured property({}), and qualifier({}). Total paths counted={}".format(self.me.getTotal(),self.mp.getTotal(),self.q.getTotal(),total),label="mp1", elements="|X|x|x|x|x|", length = "400pt"))
        print("\hline\nPattern&ME-freq&MP-freq&Q-freq&Total\\\\\n\\hline")
        count = 0
        for x in self.merged.keys():
            count+=1
            if count == maxCount:
                break
            print("{}&{}&{}&{}&{}\\\\\n\\hline".format(",".join(list(x)),self.merged[x]["me"],self.merged[x]["mp"],self.merged[x]["q"],self.merged[x]["total"]))
        print(endTable())
        print("\\clearpage")





        
