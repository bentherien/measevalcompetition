import torch.nn as nn
import torch
import torch.nn.functional as F

#In-house (CLaC) layers for graphical models



class GCNN(torch.nn.Module):
    #An impelimenaiton of Graph Convolutional network
    #By: Parsa Bagherzadeh
    #Date: July 2020
    #CLaC Lab

    def __init__(self, d_model,dependency_types):
        super(GCNN, self).__init__()
        self.label_list=[['self',nn.Linear(d_model,d_model)]]

        primes=list()
        for ele in dependency_types:
            primes.append(ele+'p')
        dependency_types=dependency_types+primes

        for label in dependency_types:
            self.label_list.append([label,nn.Linear(d_model,d_model)])
        self.weights=nn.ModuleDict(self.label_list)

    def forward(self,input,dependency_triples):
        #input shape must be T*d
        #T: number of tokens
        #d: dimensionality of token represenations

        h_l=input
        tensor_dict={}
        for i in range(len(input)):
            tensor_dict[i]=[self.weights['self'](h_l[i])]

        for triple in dependency_triples:
            tensor_dict[triple[0]].append(self.weights[triple[1]](h_l[triple[2]]))
            tensor_dict[triple[2]].append(self.weights[triple[1]+'p'](h_l[triple[0]]))

        h_l_plus1=list()
        for i in range(len(input)):
            h_l_plus1.append(torch.stack(tensor_dict[i]).sum(dim=0))
        h_l_plus1=F.relu(torch.stack(h_l_plus1))

        return h_l_plus1

dep = ["obj","subj"]
a = GCNN(100,dep)




