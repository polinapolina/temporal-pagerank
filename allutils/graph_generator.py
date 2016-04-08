__author__ = 'Polina'
import networkx as nx
import copy
import numpy as np
from datetime import datetime, timedelta
import os.path
import matplotlib.pyplot as plt

def getToy():
    #G = nx.Graph()
    #G.add_edges_from([(1,2,{'weight': 0.25}), (2,3, {'weight': 0.25})])
    G = nx.DiGraph()
    G.add_edges_from([(1,2,{'weight': 1.0}), (3,2, {'weight': 1.0})])
    nrm = float(sum(G.degree(weight = 'weight').values()))
    for i in G.edges_iter(data=True):
        G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm
    return G

def getSubgraph(G, N = 1000):
    Gcc = sorted(nx.connected_component_subgraphs(G.to_undirected()), key = len, reverse=True)
    print len(Gcc)
    nodes = set()
    i = 0

    while len(nodes) < N:
        s = np.random.choice(Gcc[i].nodes())
        i += 1
        nodes.add(s)
        for edge in nx.bfs_edges(G.to_undirected(), s):
            nodes.add(edge[1])
            if len(nodes) == N:
                break
    return nx.subgraph(G, nodes)


def getGraph(edgesTS):
    G = nx.DiGraph()
    edges = {}

    for item in edgesTS:
        edge = item[1]
        edges[edge] = edges.get(edge, 0.0) + 1.0

    #nrm = float(sum(edges.values()))
    G.add_edges_from([(k[0],k[1], {'weight': v}) for k,v in edges.iteritems()])
    #G.add_edges_from([tuple(edge)])
    return G

def readRealGraph(filepath):
    edgesTS = []
    nodes = set()
    edges = set()
    lookup = {}
    c = 0
    with open(filepath,'r') as fd:
        for line in fd.readlines():

            line = line.strip()
            items = line.split(' ')
            tstamp = ' '.join(items[0:2])
            tstamp = tstamp[1:-1]
            tstamp = datetime.strptime(tstamp, '%Y-%m-%d %H:%M:%S')
            t = items[2:4]
            t = map(int,t)
            if t[0] == t[1]:
                continue
            #t.sort(); #undirected

            if tuple(t) in lookup.keys():
                num = lookup[tuple(t)]
            else:
                num = c
                lookup[tuple(t)] = c
                c += 1
            edgesTS.append((tstamp, tuple(t), num ))
            nodes.add(t[0])
            nodes.add(t[1])
            edges.add(tuple([t[0],t[1]]))
    fd.close()
    return edgesTS, nodes, edges

def weighted_DiGraph(n, seed = 1.0, mode='random', weights='random'):
    if mode == 'ER':
        G = nx.erdos_renyi_graph(n, p=0.1, directed=True, seed = seed)
    elif mode == 'PL':
        G = nx.scale_free_graph(n*10, seed=seed)
        G = nx.DiGraph(G)
        G.remove_edges_from(G.selfloop_edges())
    elif mode == 'BA':
        G = nx.barabasi_albert_graph(n, 3, seed=None)
        G = nx.DiGraph(G)
    elif mode == 'random':
        G = nx.scale_free_graph(n)
        G = nx.DiGraph(G)
        G.remove_edges_from(G.selfloop_edges())
    else:
        edgesTS, _, _ = readRealGraph(os.path.join('.','..',"Data", mode+".txt"))
        G = getGraph(edgesTS)
        G = nx.DiGraph(G)
        G.remove_edges_from(G.selfloop_edges())
        G = getSubgraph(G, n)
    

    for i in G.nodes_iter():
        if G.out_degree(i) == 0:
            for j in G.nodes_iter():
                if i != j:
                    G.add_edge(i, j, weight=1.0)

    #nx.draw(G)
    #plt.show()


    # flag = True
    # while flag:
    #     nodes = copy.deepcopy(G.nodes())
    #     flag = False
    #     for i in nodes:
    #         #if not G.neighbors(i) and not G.predecessors(i):
    #         if not G.neighbors(i) or not G.predecessors(i):
    #             G.remove_node(i)
    #             flag = True

    print nx.info(G)


    if weights == 'random':
        w = np.random.uniform(1e-5, 1.0, G.number_of_edges())
        w /= sum(w)
        c = 0
        for i in G.edges_iter():
            G[i[0]][i[1]]['weight'] = w[c]
            c += 1
    elif weights == 'uniform':
        w = 1.0/G.number_of_edges()
        for i in G.edges_iter():
            G[i[0]][i[1]]['weight'] = w
    else:
        nrm = float(sum(G.out_degree(weight = 'weight').values()))
        for i in G.edges_iter(data=True):
            G[i[0]][i[1]]['weight'] = i[-1]['weight']/nrm
    return G

def change_weights(G):
    #w = np.random.uniform(1e-5, 1.0, G.number_of_edges())
    w = np.random.uniform(0.0, 1.0, G.number_of_edges())
    w /= sum(w)
    c = 0
    for i in G.edges_iter():
        G[i[0]][i[1]]['weight'] = w[c]
        c += 1
    return G
