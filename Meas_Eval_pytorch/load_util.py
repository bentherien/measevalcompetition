class Sample:
    tokens=None
    target=None


    def __init__(self):
        self.tokens=list()
        self.target = list()




def load_sample(fname):
    file=open(fname,'r',encoding="utf-8")

    all_samples=list()

    all_str=file.read()

    all_str=all_str.split('\n\n')

    for sample in all_str:
        new_sample=Sample()

        tokens=sample.split('\n')
        append = True

        for token in tokens:
            if token != "":
                new_sample.tokens.append(token.split('\t')[0])
                new_sample.target.append(token.split('\t')[1])
            else:
                append = False
                
        if append:
            all_samples.append(new_sample)

    return all_samples






