import allutils.graph_generator
from allutils.general_PR import flowPR
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import uuid
from allutils.utils_experiments import *


if __name__ == "__main__":

    #mode = sys.argv[1]
    mode = 'facebook'
    
    n = 100 #number of nodes in the graph
    iters = 100000 #number of temporal edges in the graph
    
    gamma = 1.0

    if mode != 'random':
        weights = 'real'
    else:
        weights = 'random'
        
    colors = ['k', 'r', 'b', 'g']
    styles = ['-', '--', ':', '-.']
    betas = [0.0, 0.1, 0.5, 0.9]
    
    plt.rcParams.update({'font.size': 20, 'lines.linewidth': 3})
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 25
    plt.figure('beta')
    
    for i in xrange(len(betas)):
        beta = betas[i]
        G = allutils.graph_generator.weighted_DiGraph(n, seed = 1.0, mode = mode, weights = weights)
        norm = sum(G.out_degree(weight='weight').values())
        sampling_edges = {e[:-1]: e[-1]['weight']/norm for e in G.edges_iter(data=True)}
        stream = [sampling_edges.keys()[i] for i in np.random.choice(range(len(sampling_edges)), size=iters, p=sampling_edges.values())]
        personalization = {k: v / norm for k, v in G.out_degree(weight='weight').iteritems()}
        p_prime_nodes = {i: personalization[i]/G.out_degree(i, weight='weight') for i in G.nodes_iter()}
        pr_basic = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
        
        RS, current = {}, {}
        RS, current, tau, spearman, pearson, error, x = flowPR(p_prime_nodes, pr_basic, stream, RS, current, iters = iters, beta = beta, gamma = gamma)

        plt.plot(pearson, color=colors[i], linestyle = styles[i])
    
    

    leg = []
    for i in betas:
        leg += ['beta='+str(i)]

    plt.legend(leg, loc=0)
    plt.ylim((0, 1.0))
    plt.xlabel('number of temporal edges', fontsize=25)
    plt.tight_layout()
    plt.savefig(mode+'_beta_pearson.pdf')