import networkx as nx

def getShortestPath(source,target,dependencies):
    edges = {}
    graph = nx.DiGraph()
    for child,dep,gov in dependencies:
        graph.add_node(child)
        graph.add_node(gov)

        graph.add_edge(child,gov)
        edges[(child,gov)] = "r"+dep

        graph.add_edge(gov,child)
        edges[(gov,child)] = dep

    try:
        sp = nx.shortest_path(graph, source=source, target=target)
    except nx.exception.NodeNotFound:
        print("error, invalid source or target")
        return 

    edgePath = [(sp[i],sp[i+1],) for i in range(0,len(sp)-1)]
    
    return [edges[x] for x in edgePath]