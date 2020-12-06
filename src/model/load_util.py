class Sample:
    tokens=None
    target=None

    def __init__(self):
        self.tokens=list()
        self.target = list()
        self.pos=list()
        self.deps=list()



def load_sample_dep(fname):
    all_str = None
    with open(fname,'r',encoding="utf-8") as tempFile:
        all_str = tempFile.read().split("\n\n")

    all_samples = []
    for sample in all_str:
        new_sample = Sample()

        tokens = sample.split('\n')
        dep = {}
        for token in tokens:
            if token != "":
                split = token.split('\t')
                new_sample.tokens.append(split[0])
                new_sample.target.append(split[1])
                new_sample.pos.append(split[2])
                if split[4].lower() not in ["root",""," "]:
                    try:
                        dep[int(split[3])].append(split[4])
                    except KeyError:
                        dep[int(split[3])] = [split[4]]
                    try:
                        dep[int(split[5])].append("r-"+split[4])
                    except KeyError:
                        dep[int(split[5])] = ["r-"+split[4]]
        
        for x in range(len(new_sample.tokens)):
            try:
                new_sample.deps.append(dep[x])
            except KeyError:
                new_sample.deps.append([])
        
        if new_sample.tokens != []:
            all_samples.append(new_sample)

    return all_samples
            

    



def load_sample(fname):
    file=open(fname,'r',encoding="utf-8")

    all_samples=list()

    all_str=file.read()

    all_str=all_str.split('\n\n')

    for sample in all_str:
        new_sample=Sample()

        tokens=sample.split('\n')

        for token in tokens:
            if token != "":
                new_sample.tokens.append(token.split('\t')[0])
                new_sample.target.append(token.split('\t')[1])
                new_sample.pos.append(token.split('\t')[2])
	
        i=-1
        flag = 0
        prev=None
        for target in new_sample.target:
            i=i+1
            if target!='o':
                if target!=prev:
                    flag=0
                if flag==0:
                    prev=target
                    target='B-'+target
                    new_sample.target[i]=target
                    flag=1
                elif flag==1:
                    target = 'I-' + target
                    new_sample.target[i] = target
            else:
                flag=0






        all_samples.append(new_sample)

    return all_samples