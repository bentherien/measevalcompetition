import numpy as np


def expand_annotations(tokens,target,tokenizer):
    i=-1
    new_annotations=list()
    for token in tokens:
        i=i+1

        tokenized=tokenizer.tokenize(token)

        if len(tokenized)==1:
            new_annotations.append(target[i])
        else:
            new_annotations=new_annotations+len(tokenized)*[target[i]]

    return new_annotations



def expand_tokens(tokens,tokenizer):
    new_tokens=list()
    for token in tokens:
        new_tokens=new_tokens+tokenizer.tokenize(token)
    return new_tokens



def contract_annotations(tokens,target,tokenizer):

    new_annotations=list()

    for token in tokens:
        cur_list=list()
        tokenized=tokenizer.tokenize(token)

        for i in range(len(tokenized)):
            cur_list.append(target[0])
            target.pop(0)
        if len(cur_list)==0:
            aa=2
        new_annotations.append(cur_list)
    return new_annotations


def merge_annotations(annotations):

    new_annotations=list()

    for li in annotations:
        if len(li)==1:
            new_annotations.append(li[0])
        else:
            if len(li)==0:
                aa=2
            most_frq=max(set(li), key = li.count)
            new_annotations.append(most_frq)


    return new_annotations



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
    modif=None
    doc_id=None

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
        self.modif=list()

def load_samples(fname):

    file=open(fname+'.tsv')

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
                new_sample.dep.append((int(token[3]),token[4],int(token[5])))

        annotations=sample[1].split('\n')
        qas=annotations[0]
        mes = annotations[1]
        mps = annotations[2]
        qls = annotations[3]

        qa_me_rel = annotations[4]
        qa_mp_rel = annotations[5]
        mp_me_rel = annotations[6]
        qa_ql_rel = annotations[7]

        modifs = annotations[8]


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


        modif_dict=dict()

        if len(modifs.split('\t'))>1:
            for modif in modifs.split('\t')[1:]:
                modif=modif.split(':')
                if int(modif[0]) not in modif_dict:
                    modif_dict[int(modif[0])]=[modif[1]]
                else:
                    modif_dict[int(modif[0])]=modif_dict[int(modif[0])]+[modif[1]]


            for key in modif_dict:
                new_sample.modif.append((key, modif_dict[key], new_sample.qa[key][0], new_sample.qa[key][1]))




        all_samples.append(new_sample)
    return all_samples


import networkx as nx


def getShortestPath(source, target, dependencies):
    edges = {}
    graph = nx.DiGraph()
    for child, dep, gov in dependencies:
        graph.add_node(child)
        graph.add_node(gov)

        graph.add_edge(child, gov)
        edges[(child, gov)] = "r" + dep

        graph.add_edge(gov, child)
        edges[(gov, child)] = dep

    try:
        sp = nx.shortest_path(graph, source=source, target=target)
    except nx.exception.NodeNotFound:
        print("error, invalid source or target")
        return

    edgePath = [(sp[i], sp[i + 1],) for i in range(0, len(sp) - 1)]

    return [edges[x] for x in edgePath]




def construct_graph(dependencies):
    edges = {}
    graph = nx.DiGraph()
    for child, dep, gov in dependencies:
        graph.add_node(child)
        graph.add_node(gov)

        graph.add_edge(child, gov)
        edges[(child, gov)] = "r" + dep

        graph.add_edge(gov, child)
        edges[(gov, child)] = dep

    return graph, edges


def ShortestPath(source,target,graph,edges):
    try:
        sp = nx.shortest_path(graph, source=source, target=target)
    except nx.exception.NodeNotFound:
        print("error, invalid source or target")
        return

    edgePath = [(sp[i], sp[i + 1],) for i in range(0, len(sp) - 1)]

    return [edges[x] for x in edgePath]



def inInterval(key,inetrval):
    if key>=inetrval[0] and key<=inetrval[1]:
        return True
    else:
        return False


from random import random
def add_path_to_samples(all_samples):
    for sample in all_samples:
        if len(sample.qa)>0:
            graph, edges = construct_graph(sample.dep)
            for rel in sample.qa_mp_rel:
                qa_span=sample.qa[rel[0]]
                mp_span = sample.mp[rel[1]]

                qa_pos=sample.pos[qa_span[0]:qa_span[1]+1].copy()

                if 'CD' in qa_pos:
                    last_number=len(qa_pos) - 1 - qa_pos[::-1].index('CD') + qa_span[0]
                elif 'JJ' in qa_pos:
                    last_number=len(qa_pos) - 1 - qa_pos[::-1].index('JJ') + qa_span[0]
                else:
                    last_number=qa_span[1]
                mp_pos=sample.pos[mp_span[0]:mp_span[1]+1].copy()

                i=-1
                for tag in mp_pos:
                    i=i+1
                    if tag in ['NN','NNS','NNP','NNPS']:
                        mp_pos[i]='NN'

                if 'NN' in mp_pos:
                    last_noun=len(mp_pos) - 1 - mp_pos[::-1].index('NN') + mp_span[0]

                elif 'JJ' in mp_pos:
                    last_noun = len(mp_pos) - 1 - mp_pos[::-1].index('JJ') + mp_span[0]

                else:
                    last_noun=mp_span[1]



                sample.neg_pos_path.append((last_number,last_noun,ShortestPath(last_number,last_noun,graph,edges),1,qa_span))



            for rel in sample.qa_me_rel:
                qa_span=sample.qa[rel[0]]
                me_span = sample.me[rel[1]]

                qa_pos = sample.pos[qa_span[0]:qa_span[1] + 1].copy()

                if 'CD' in qa_pos:
                    last_number = len(qa_pos) - 1 - qa_pos[::-1].index('CD') + qa_span[0]
                elif 'JJ' in qa_pos:
                    last_number = len(qa_pos) - 1 - qa_pos[::-1].index('JJ') + qa_span[0]
                else:
                    last_number = qa_span[1]

                me_pos = sample.pos[me_span[0]:me_span[1] + 1].copy()

                i = -1
                for tag in me_pos:
                    i = i + 1
                    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                        me_pos[i] = 'NN'

                if 'NN' in me_pos:
                    last_noun = len(me_pos) - 1 - me_pos[::-1].index('NN') + me_span[0]

                elif 'JJ' in me_pos:
                    last_noun = len(me_pos) - 1 - me_pos[::-1].index('JJ') + me_span[0]

                else:
                    last_noun = me_span[1]


                sample.neg_pos_path.append((last_number, last_noun, ShortestPath(last_number, last_noun, graph, edges), 2,qa_span))



            for rel in sample.mp_me_rel:
                mp_span=sample.mp[rel[0]]
                me_span = sample.me[rel[1]]


                mp_pos = sample.pos[mp_span[0]:mp_span[1] + 1].copy()

                i = -1
                for tag in mp_pos:
                    i = i + 1
                    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                        mp_pos[i] = 'NN'

                if 'NN' in mp_pos:
                    last_noun = len(mp_pos) - 1 - mp_pos[::-1].index('NN') + mp_span[0]

                elif 'JJ' in mp_pos:
                    last_noun = len(mp_pos) - 1 - mp_pos[::-1].index('JJ') + mp_span[0]

                else:
                    last_noun = mp_span[1]




                me_pos = sample.pos[me_span[0]:me_span[1] + 1].copy()

                i = -1
                for tag in me_pos:
                    i = i + 1
                    if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
                        me_pos[i] = 'NN'

                if 'NN' in me_pos:
                    last_noun2 = len(me_pos) - 1 - me_pos[::-1].index('NN') + me_span[0]

                elif 'JJ' in me_pos:
                    last_noun2 = len(me_pos) - 1 - me_pos[::-1].index('JJ') + me_span[0]

                else:
                    last_noun2 = me_span[1]

                sample.neg_pos_path.append((last_noun, last_noun2, ShortestPath(last_noun, last_noun2, graph, edges), 3, mp_span))


            neg_tokens=list()

            for i in range(len(sample.tokens)):
                if sample.pos[i] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    neg_tokens.append(i)
            candidate_pairs=list()
            for path in sample.neg_pos_path:
                for token_idx in neg_tokens:
                    if inInterval(token_idx,path[4]):
                        continue
                    if (path[0],token_idx) not in candidate_pairs:
                        candidate_pairs.append((path[0],token_idx))

            for path in sample.neg_pos_path:
                if (path[0],path[1]) in candidate_pairs:
                    candidate_pairs.pop(candidate_pairs.index((path[0],path[1])))

            for pair in candidate_pairs:
                sample.neg_pos_path.append((pair[0], pair[1], ShortestPath(pair[0], pair[1], graph, edges), 0,None))

    return all_samples


def modify_spans(ptr,l,all_spans):
    for index,span in enumerate(all_spans):
        if span[0]>ptr:
            all_spans[index][0]=all_spans[index][0]+l-1
            all_spans[index][1] = all_spans[index][1] + l - 1
        elif inInterval(ptr,span):
            all_spans[index][1]=all_spans[index][1]+l-1

    return all_spans
def expand_modif_spans(tokens,modifs,tokenizer):
    all_spans=list()
    for instance in modifs:
        all_spans.append([instance[2],instance[3]])
    temp_tokens=list()
    ptr=0
    for token in tokens:
        new_tokens=tokenizer.tokenize(token)
        temp_tokens=temp_tokens+(tokenizer.tokenize(token))
        if len(new_tokens)>1:
            all_spans=modify_spans(ptr,len(new_tokens),all_spans)

        ptr=ptr+len(new_tokens)

    return all_spans


def get_qa_spans(predictions):
    spans=list()
    state=0
    i=-1
    for pred in predictions:
        i=i+1
        if state==0:
            if pred=='o' or pred=='MP' or pred=='ME' or pred=='QL':
                continue
            else:
                state=1
                spans.append(i)
                continue
        elif state==1:
            if pred!='o' and pred!='MP' and pred!='ME' and pred!='QL':
                continue
            else:
                state=0
                spans.append(i-1)
                continue

    if len(spans)%2!=0:
        spans.append(i)
    i=-1
    all_spans=list()
    cur_span=list()
    for ofs in spans:
        i=i+1
        if i%2==0:
            cur_span.append(ofs)
        else:
            cur_span.append(ofs)
            all_spans.append(cur_span)
            cur_span=list()

    return all_spans


def Var(v):
    if CUDA:
        return Variable(v.cuda())
    else:
        return Variable(v)


def id_to_tag(seq, to_tag):
    res = list()

    for ele in seq:
        res.append(to_tag[ele])

    return res


def convert_2_tensor(seq, to_ix, dt):
    if to_ix == None:
        return Var(torch.tensor(seq, dtype=dt))
    else:
        idxs = list()
        for w in seq:
            if w in to_ix:
                idxs.append(to_ix[w])
        return Var(torch.tensor(idxs, dtype=dt))


def predict_multi_class(class_score, CUDA):
    if CUDA:
        class_score = class_score.cpu().detach().numpy()
    else:
        class_score = class_score.detach().numpy()
    predictions = list()
    for instance in class_score:
        current_pred = [1 if a_ >= 0.5 else 0 for a_ in instance]
        current_pred = [i for i, e in enumerate(current_pred) if e != 0]
        predictions.append(current_pred)
    return predictions


def predict_class(class_score, CUDA):
    if CUDA:
        class_score = class_score.cpu().detach().numpy()
    else:
        class_score = class_score.detach().numpy()
    classes = list()
    for seq in class_score:
        classes.append(np.argmax(seq))
    return classes

def id_to_tag_multi_class(pred, inv_dict):
    all_pred = list()
    for instance in pred:
        cur_pred = list()
        for ele in instance:
            cur_pred.append(inv_dict[ele])
        all_pred.append(cur_pred)
    return all_pred
