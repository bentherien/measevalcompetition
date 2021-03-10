



class PathData:

    class Sentence:
        def __init__(self,sentData):
            try:
                self.sentence = sentData["sentence"]
            except KeyError:
                self.sentence = []
            sentData.pop('sentence', None)
            self.orig_paths = sentData
            self.samples = [[y["path"] for y in x] for x in self.orig_paths.values()]
            self.targets = [[y["target"] for y in x] for x in self.orig_paths.values()] 

    # def getTags(self):
    #     tags = []
    #     for x in self.sents:
    #         for targetList in x.targets:
    #             for target in targetList:
    #                 tags.append(target)
    #     return tags

    # def getDeps(self):
    #     deps = []
    #     for x in self.sents:
    #         for sentDepList in x.samples:
    #             for depList in sentDepList:
    #                 for dep in depList:
    #                     deps.append(dep)
    #     return deps


    # def getDeps2(self):
    #     deps = []
    #     for x in self.sents:
    #         for sentDepList in x.samples:
    #             for depList in sentDepList:
    #                 for dep in depList:
    #                     deps.append(dep[0])
    #     return deps


    class Sent:
        def __init__(self,sentData):
            self.sentData = sentData
            self.path = sentData["path"][1:]
            self.target = sentData["target"]

    def getTags(self):
        tags = []
        for x in self.sents:
            tags.append(x.target)
        return tags



    def getDeps2(self):
        deps = []
        for x in self.sents:
            for depList in x.path:
                    for dep in depList:
                        deps.append(dep[0])
        return deps



    def __init__(self, data):
        self.sents = [PathData.Sent(x) for x in data]

    
class BertPathData:
    class Node:
        def __init__(self,path,quant,noun,target):
            self.path = [x[0] for x in path][1:]
            self.qSpan = quant
            self.nSpan = noun
            self.target = target

    def __init__(self,data):
        self.data = [BertPathData.Node(x["path"],x["quantSpan"],x["nounSpan"],x["target"]) for x in data]



