



class PathData:

    class Sentence:
        def __init__(self,sentData):
            self.sentence = sentData["sentence"]
            sentData.pop('sentence', None)
            self.orig_paths = sentData
            self.samples = [[y["path"] for y in x] for x in self.orig_paths.values()]
            self.targets = [[y["target"] for y in x] for x in self.orig_paths.values()] 

    def getTags(self):
        tags = []
        for x in self.sents:
            for targetList in x.targets:
                for target in targetList:
                    tags.append(target)
        return tags

    def getDeps(self):
        deps = []
        for x in self.sents:
            for sentDepList in x.samples:
                for depList in sentDepList:
                    for dep in depList:
                        deps.append(dep)
        return deps


    def getDeps2(self):
        deps = []
        for x in self.sents:
            for sentDepList in x.samples:
                for depList in sentDepList:
                    for dep in depList:
                        deps.append(dep[0])
        return deps


    def __init__(self, data):
        self.sents = [PathData.Sentence(x) for x in data]

    



