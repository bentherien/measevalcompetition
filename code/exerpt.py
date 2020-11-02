from code.common import LATEST_tsv

class exerpt:
    """
    Class exerpt
    Description: a simple class to contain data for the measeval competition

    self.name : the measeval given name associated to the document
    self.txt : the raw text of the document
    self.ann : the brat annotations of the document(deprecated)
    self.tsv : a pandas dataframe containing all the tab seperated value data
    self.grobid : grobid quantities json output for quantity detection
    self.doc : The spacy doc generated from processing on this particular document
    """
    
    def __init__(self, name, txt, ann, tsv, grobid, nlp):
        self.name = name
        self.txt = txt
        self.ann = ann
        self.tsv = tsv
        self.grobid = grobid
        self.context = False
        self.posTag = False
        global LATEST_tsv
        LATEST_tsv = self.tsv
        self.doc = nlp(self.txt)
        self.hashFromFrame()
        
    def getContext(self, count = 3):
        if self.context == False: 
            beforeContext = []
            afterContext = []
            self.context = True
            for start,end in zip(self.tsv["startOffset"].values, self.tsv["endOffset"].values ):#self.tsv.iterrows():
                a = getIndexSeperated(exerpt.txt[0],end,count = count)
                b = getIndexSeperated(exerpt.txt[0],start,count = count,forward = False)
                beforeContext.append(self.txt[0][b:start])
                afterContext.append(self.txt[0][end:a])
            self.tsv.insert(3,"beforeContext", beforeContext)
            self.tsv.insert(3,"afterContext", afterContext)
            
    def hashFromFrame(self):
        self.measurements = {}
        count = 0
        for index, row in LATEST_tsv.iterrows():
            if(row["annotType"] == "Quantity"):
                count+=1
                
        
            #check for creating a new dict
            try:
                if(type(self.measurements[f"Annotation{count}"]) == type(dict)):
                    continue
            except KeyError:
                self.measurements[f"Annotation{count}"] = {}

            
            self.measurements[f"Annotation{count}"][row["annotType"]] = {
                "startOffset" : row["startOffset"],
                "endOffset" : row["endOffset"],
                "annotSet" : row["annotSet"],
                "annotType" : row["annotType"],
                "annotId" : row["annotId"],
                "text" : row["text"],
                "other" : row["other"],
            }

            
    def getPosTag(self):
        if self.context and self.posTag == False: 
            bTag = []
            aTag = []
            self.posTag = True
            for before, after in zip(self.tsv["beforeContext"].values,self.tsv["afterContext"].values):
                bTag.append(nltk.pos_tag(word_tokenize(before)))
                aTag.append(nltk.pos_tag(word_tokenize(after)))
            self.tsv.insert(3,"beforeTag", bTag)
            self.tsv.insert(3,"afterTag", aTag)