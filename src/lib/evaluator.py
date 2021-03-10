import os
import subprocess


class Evaluator:
    

    def __init__(self,path,submissionDir):
        """
        Class to encapsulate the functionality necessary
         to automatically run tests of measeval.
        """

        if os.path.isdir(path):
            self.path = path
        else: 
            print("Incorrect path {} passed to Evaluator".format(path))

        if os.path.isdir(os.path.join(path,submissionDir)):
            self.sub = submissionDir
        else: 
            print("Incorrect path {} passed to Evaluator".format(os.path.join(path,submissionDir)))

        self.setup = False


    def setupData(self,eval=False):
        """
        Creates a directory with copies of the gold data that match the 
        document ids of the created documents from the submission directory
        """ 
        if os.path.isdir(os.path.join(self.path,"tempGold")):
            os.system("rm -r {}".format(os.path.join(self.path,"tempGold")))

        os.mkdir(os.path.join(self.path,"tempGold"))

        if eval:
            for x in os.listdir(os.path.join(self.path,"eval")):
                os.system("cp {} {}".format(os.path.join(self.path,"gold",x), os.path.join(self.path,"tempGold")))
        else:
            for x in os.listdir(os.path.join(self.path,self.sub)):
                os.system("cp {} {}".format(os.path.join(self.path,"gold",x), os.path.join(self.path,"tempGold")))

        self.setup = True


    def testRegular(self, complete=False, latex=False):
        if not self.setup:
            self.setupData()
        
        cmd = "/usr/bin/python3 measeval-eval.py -i {}/ -s {}/ -g tempGold/".format(self.path,self.sub)
        try:
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
        results = str(result).split(":)")[-1]
        
        keys = [
            "Overall Score Exact Match:",
            "Overall Score F1 (Overlap):",
            "Precision:",
            "Recall:",
            "F-measure:"
        ]
        d = {}
        for x in results.split("\\n"):
            for y in keys:
                if y in x: 
                    d[y] = x[len(y):]

        if latex:
            return d

        
        if complete:
            for x in results:
                print(x)
        else:
            for key in d: 
                print(key,d[key])

        return d



    def testClass(self, complete=False, latex=False):
        if not self.setup:
            self.setupData()
        
        cmd = "/usr/bin/python3 measeval-eval.py -i {}/ -s {}/ -g tempGold/ -m {}".format(self.path,self.sub,"class")
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        results = str(result).split(":)")[-1]
        results = results.split("\\n")
        results = [x.split("\n")[0] for x in results]

        keys = [
            "Gold count of Quantity:",
            "Gold count of MeasuredProperty:",
            "Gold count of MeasuredEntity:",
            "Gold count of Qualifier:",
            "Submission count of Quantity:",
            "Submission count of MeasuredProperty:",
            "Submission count of MeasuredEntity:",
            "Submission count of Qualifier:",
            "F1 (Overlap) Score for Quantity:",
            "F1 (Overlap) Score for MeasuredProperty:",
            "F1 (Overlap) Score for MeasuredEntity:",
            "F1 (Overlap) Score for Qualifier:",
            "F1 (Overlap) Score for Unit:",
            "F1 (Overlap) Score for modifier:",
            "F1 (Overlap) Score for HasQuantity:",
            "F1 (Overlap) Score for HasProperty:",
            "F1 (Overlap) Score for Qualifies:",
        ]

        d = {}
        for x in results:
            for y in keys:
                if y in x: 
                    d[y] = x[len(y):]


        if complete:
            for x in results:
                print(x)
        else:
            for key in d: 
                print(key,d[key])

        print()

        if latex:
            reg = self.testRegular(latex=True)
            print(reg)

            overall = reg["Overall Score F1 (Overlap):"]
            l = [reg['Overall Score Exact Match:'],reg['Precision:'],reg['Recall:'],reg['F-measure:']]
            l = [str(round(float(x),3))[1:] for x in l]
            scores = [overall] + list(d.values())[8:]
            scores = [str(round(float(x),3))[1:] for x in scores]
            print("EM&P&R&F-Measure&overall&QA&ME&MP&QL&Unit&Modifier&HasQA&HasProp&Quals\\%")
            #print("&".join(scores),"\\\\%","&".join(l))
            acc = l + scores 
            print("&".join(acc),"\\\\")
        

        
