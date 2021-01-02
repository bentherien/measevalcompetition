
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt






class Graph:

    def addToken(self,node):
        """
        A helper method to add an annotated token to one of the lists 
        me, mp, ql, qa
        """
        annotIdList = node[1]._.other["annotID"]
        print(annotIdList)
        annotType = node[1]._.all
        for annotId in annotIdList:
            try:
                self.annotNodes[annotId]
            except KeyError:
                self.annotNodes[annotId] = {}

            try: 
                self.annotNodes[annotId][annotType].append(node)
            except KeyError:
                self.annotNodes[annotId][annotType] = [node]

    def __init__(self,sentence):
        """
        annotNodes is a dictionary of dictionaries that contains all the different
        annotations of the sentence seperated by group
        """
        self.annotNodes = {}

            
        self.edges = {}
        self.graph = nx.DiGraph()
        for token in sentence:
            tokNode = (token.text,token,token.dep_,token.i,)
            self.graph.add_node(tokNode)
            if token._.all != "o":
                self.addToken(tokNode)

            for child in token.children:
                childNode = (child.text,child,child.dep_,child.i,)
                self.graph.add_node(childNode)

                self.graph.add_edge(tokNode,childNode)
                self.edges[(tokNode,childNode,)] = child.dep_

                self.graph.add_edge(childNode,tokNode,)
                self.edges[(childNode,tokNode)] = "r"+child.dep_


    def getSource(tokens):
        if tokens[-1][1].tag_ == "CD":
            return tokens[-1]

        sourceCD = None
        sourceJJ = None
        for x in tokens: 
            if x[1].tag_ == "CD":
                sourceCD = x
            elif x[1].tag_ == "JJ":
                sourceJJ = x

        if sourceCD != None:
            return sourceCD
        elif sourceJJ !=  None:
            return sourceJJ
        else:
            print("in Graph class method getSource, compromise was made for {} with pos {}".format(tokens[-1][1].text,tokens[-1][1].tag_))
            return tokens[-1]


    def getTarget(tokens):
        """
        Attemps to find the last noun in the span and return it, if no noun is found, return last token
        """
        for x in reversed(tokens):
            if(x[1].pos_ in ["NOUN","PROPN"]):
                return x

        return tokens[-1]


    def pathToDigraph(self,nodes,edges):
        graph = nx.DiGraph()
        e = {}
        for i in range(len(nodes)-2):
            graph.add_node(nodes[i])
            graph.add_node(nodes[i+1])
            graph.add_edge(nodes[i],nodes[i+1])
            graph.add_edge(nodes[i+1],nodes[i])
            e[(nodes[i],nodes[i+1],)] = edges[(nodes[i],nodes[i+1],)]
            e[(nodes[i+1],nodes[i],)] = edges[(nodes[i+1],nodes[i],)]
        return graph, e

    def drawToFile(self,graph,edgeDict,dist=5,filename=''):
        if graph.edges == tuple():
            return 

        pos = nx.spring_layout(
            graph,
            k=dist/np.sqrt(len(graph.nodes)), 
            pos=None, 
            fixed=None, 
            iterations=100, 
            threshold=0.00001, 
            weight='weight', 
            scale=10, 
            center=None, 
            dim=2, 
            seed=None
        )
        #pos = nx.rescale_layout_dict(pos,5000)
        #pos = nx.rescale_layout(pos,d)
        plt.figure(figsize=(20,20))

        nx.draw(
            graph,
            pos,
            edge_color='black',
            width=1,
            linewidths=1,
            font_size=16,
            node_size=2000,
            node_color='pink',
            alpha=0.9,
            labels= {node:(node[0],node[1]._.all) for node in graph.nodes()}
        )

        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edgeDict,
            label_pos=0.5,
            font_size=20, 
            font_color='red', 
            font_family='sans-serif', 
            font_weight='normal', 
            alpha=1.0, 
            bbox=None, 
            ax=None, 
            rotate=True, 
        )

        plt.axis('off')

        if filename != '':
            plt.savefig('filename')

        plt.show()


    def draw(self,filename=''):
        self.drawToFile(self.graph,self.edges,dist=5,filename=filename)



    def getShortesPathFromQuant(self, target="ME", getSource=getSource, getTarget=getTarget, draw=False,distance=2):
        if target not in ["ME","MP","QL"]:
            print("Invalid target specified")
            return None

        paths = []
        nodePaths = []
        for annot in self.annotNodes.values():
            #iterate over each annotation dict
            if target not in annot:
                continue

            if "QA" not in annot:
                print("nothing",annot)
                continue

            target = getTarget(annot[target])
            source = getSource(annot["QA"])
            sp = nx.shortest_path(self.graph, source=source, target=target)
            nodePaths.append(sp)
            edgePath = [(sp[i],sp[i+1],) for i in range(0,len(sp)-1)]
            paths.append([self.edges[x] for x in edgePath])

        if draw == True:
            for x in nodePaths:
                g,e = self.pathToDigraph(x,self.edges)
                self.drawToFile(graph=g,edgeDict=e,dist=distance,filename='')

        return paths, nodePaths


    def getPathVisualization(self, nodePath):
        s = ""
        for i,node in enumerate(nodePath):
            s += "({},{},{})".format(node[0],node[1].tag_,node[1]._.all)
            if i < len(nodePath)-1:
                s+= "-->{}-->".format(self.edges[(nodePath[i],nodePath[i+1],)])
        return s

            



        
    
   


 
            


    
    

        