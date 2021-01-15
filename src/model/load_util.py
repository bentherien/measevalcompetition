class Sample:
    tokens=None
    pos=None
    qa=None
    dep=None
    me=None
    mp=None
    ql=None
    qa_me_rel=None
    qa_mp_rel=None
    mp_me_rel=None
    qa_ql_rel=None
    neg_pos_path=None

    targets=None
    def __init__(self):
        self.tokens=list()
        self.pos=list()
        self.dep=list()
        self.qa=dict()
        self.me=dict()
        self.mp=dict()
        self.ql=dict()
        self.qa_me_rel = list()
        self.qa_mp_rel = list()
        self.mp_me_rel = list()
        self.qa_ql_rel = list()
        self.neg_pos_path=list()
        self.targets=list()



def load_samples(fname):

    file=open(fname)

    content=file.read()

    samples=content.split('\n\n')
    all_samples=list()

    for sample in samples:
        new_sample=Sample()
        sample=sample.split('\n@@annotations@@\n')

        tokens=sample[0].split('\n')
        for token in tokens:
            token=token.split('\t')
            new_sample.tokens.append(token[0])
            new_sample.targets.append(token[1])
            new_sample.pos.append(token[2])
            if token[4]!='ROOT':
                try:
                    new_sample.dep.append((int(token[3]),token[4],int(token[5])))
                except ValueError: 
                    print((int(token[3]),token[4],int(token[5])))

        annotations=sample[1].split('\n')
        qas=annotations[0]
        mes = annotations[1]
        mps = annotations[2]
        qls = annotations[3]

        qa_me_rel = annotations[4]
        qa_mp_rel = annotations[5]
        mp_me_rel = annotations[6]
        qa_ql_rel = annotations[7]

        if len(qas.split('\t'))>1:
            for qa in qas.split('\t')[1:]:
                qa=qa.split(':')
                qa[1]=qa[1].replace('(', '').replace(')', '').split(',')
                new_sample.qa[int(qa[0])]=(int(qa[1][0]),int(qa[1][1]))
        if len(mes.split('\t'))>1:
            for me in mes.split('\t')[1:]:
                me=me.split(':')
                me[1]=me[1].replace('(', '').replace(')', '').split(',')
                new_sample.me[int(me[0])]=(int(me[1][0]),int(me[1][1]))

        if len(mps.split('\t'))>1:
            for mp in mps.split('\t')[1:]:
                mp=mp.split(':')
                mp[1]=mp[1].replace('(', '').replace(')', '').split(',')
                new_sample.mp[int(mp[0])]=(int(mp[1][0]),int(mp[1][1]))

        if len(qls.split('\t'))>1:
            for ql in qls.split('\t')[1:]:
                ql=ql.split(':')
                ql[1]=ql[1].replace('(', '').replace(')', '').split(',')
                new_sample.ql[int(ql[0])]=(int(ql[1][0]),int(ql[1][1]))

        if len(qa_me_rel.split('\t')) > 1:
            for rel in qa_me_rel.split('\t')[1:]:
                rel = rel.split(':')
                rel[1] = rel[1].replace('(', '').replace(')', '').split(',')
                new_sample.qa_me_rel.append((int(rel[1][0]), int(rel[1][1])))

        if len(qa_mp_rel.split('\t')) > 1:
            for rel in qa_mp_rel.split('\t')[1:]:
                rel = rel.split(':')
                rel[1] = rel[1].replace('(', '').replace(')', '').split(',')
                new_sample.qa_mp_rel.append((int(rel[1][0]), int(rel[1][1])))
        if len(mp_me_rel.split('\t')) > 1:
            for rel in mp_me_rel.split('\t')[1:]:
                rel = rel.split(':')
                rel[1] = rel[1].replace('(', '').replace(')', '').split(',')
                new_sample.mp_me_rel.append((int(rel[1][0]), int(rel[1][1])))

        if len(qa_ql_rel.split('\t')) > 1:
            for rel in qa_ql_rel.split('\t')[1:]:
                rel = rel.split(':')
                rel[1] = rel[1].replace('(', '').replace(')', '').split(',')
                new_sample.qa_ql_rel.append((int(rel[1][0]), int(rel[1][1])))

        all_samples.append(new_sample)
    return all_samples