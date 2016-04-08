__author__ = 'Polina'
import scipy.stats
import allutils.graph_generator
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import operator

def flowPR(p_prime_nodes, ref_pr, stream, RS, current, iters = 1000000, alpha = 0.85, beta=0.001, gamma=0.9999, normalization = 1.0, padding = 0):
    if beta == 1.0:
        beta = 0.0
        
    tau = []
    pearson = []
    spearman = []
    error = []
    x = []
    i = 0

    rank_order = [key for (key, value) in sorted(ref_pr.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_pr = np.array([ref_pr[k] for k in rank_order])

    for e in stream:
        i += 1

        RS[e[0]] = RS.get(e[0], 0.0) * gamma + 1.0 * (1.0 - alpha) * p_prime_nodes[e[0]] * normalization
        RS[e[1]] = RS.get(e[1], 0.0) * gamma + (current.get(e[0], 0.0) + 1.0 * (1.0 - alpha) * p_prime_nodes[e[0]]) * alpha * normalization
        current[e[1]] = current.get(e[1], 0.0) + (current.get(e[0], 0.0) + 1.0 * (1.0 - alpha)* p_prime_nodes[e[0]]) * alpha *(1 - beta)
        current[e[0]] = current.get(e[0], 0.0) * beta


        if (i % 100 == 0 or i == len(stream)) and len(RS) == len(ordered_pr):
            if i == iters-1:
                print sum(RS.values())
            sorted_RS4 = np.array([RS[k] / sum(RS.values()) for k in rank_order])
            tau.append(scipy.stats.kendalltau(sorted_RS4, ordered_pr)[0])
            pearson.append(scipy.stats.pearsonr(sorted_RS4, ordered_pr)[0])
            spearman.append(scipy.stats.spearmanr(sorted_RS4, ordered_pr)[0])
            error.append(np.linalg.norm(sorted_RS4 - ordered_pr))
            x.append(i+padding)

        if i == iters-1:
            print sum(RS.values())

    sorted_RS4 = np.array([RS[k] / sum(RS.values()) for k in rank_order])

    return RS, current, tau, spearman, pearson, error, x


if __name__ == "__main__":

    n = 100
    p = 0.1

    iters = 100000
    alpha = 0.85

    #beta = 0.99999
    #gamma = 0.99999

    beta = 0.0
    gamma = 1.0

    mode = 'SFree'
    #mode = 'facebook'

    weights = 'random'
    #weights = 'real'

    G = allutils.graph_generator.weighted_DiGraph(n, p, seed = 1.0, mode = mode, weights = weights)
    #nodes = G.nodes()

    # basic
    norm = sum(G.out_degree(weight='weight').values())
    sampling_edges = {e[:-1]: e[-1]['weight']/norm for e in G.edges_iter(data=True)}
    personalization = {k: v / norm for k, v in G.out_degree(weight='weight').iteritems()}
    p_prime_nodes = {i: personalization[i]/G.out_degree(i, weight='weight') for i in G.nodes_iter()}
    pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')

    rank_order = [key for (key, value) in sorted(pr.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_pr = np.array([pr[k] for k in rank_order])

    stream = [sampling_edges.keys()[i] for i in np.random.choice(range(len(sampling_edges)), size=iters, p=sampling_edges.values())]
    sorted_RS4, tau, spearman, pearson, error, epochs, top_k = flowPR_top(p_prime_nodes, pr, stream, iters = iters, beta = beta, gamma = gamma)

    for i in xrange(len(epochs)):
        plt.plot(top_k, tau[i][:])

    exit()
    print tau[-1]
    plt.plot(x, spearman, 'k-')
    plt.plot(x, pearson, 'k--')
    plt.plot(x, error, 'k:')

    # no personalization
    personalization = {k: 1.0 / G.number_of_nodes() for k in G.nodes_iter()}

    p_prime_nodes = {i: personalization[i]/S.out_degree(i, weight='weight') for i in G.nodes_iter()}
    #p_prime_edges = {e: G[e[0]][e[1]]['weight']/S[e[0]][e[1]]['weight'] for e in G.edges_iter()}

    pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
    sorted_pr = np.array([value for (key, value) in sorted(pr.items())])

    sorted_RS4, tau, spearman, pearson, error, x = flowPR(p_prime_nodes, sorted_pr, stream, iters = iters, beta = beta, gamma = gamma)
    print tau[-1]
    plt.plot(x, spearman, 'r-')
    plt.plot(x, pearson, 'r--')
    plt.plot(x, error, 'r:')

    # random personalization
    personalization = {k: np.random.uniform(1e-5, 1.0) for k in G.nodes()}
    personalization = {k: v/ sum(personalization.values()) for k,v in personalization.iteritems()}

    p_prime_nodes = {i: personalization[i]/S.out_degree(i, weight='weight') for i in G.nodes_iter()}
    #p_prime_edges = {e: G[e[0]][e[1]]['weight']/S[e[0]][e[1]]['weight'] for e in G.edges_iter()}

    pr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight='weight')
    sorted_pr = np.array([value for (key, value) in sorted(pr.items())])

    sorted_RS4, tau, spearman, pearson, error, x = flowPR(p_prime_nodes, sorted_pr, stream, iters = iters, beta = beta, gamma = gamma)
    print tau[-1]
    plt.plot(x, spearman, 'b-')
    plt.plot(x, pearson, 'b--')
    plt.plot(x, error, 'b:')
    plt.legend(['tau: degree pers.', 'corr: degree pers.', 'error: degree pers.', 'tau: no pers.', 'corr: no pers.', 'error: no pers.', 'tau: random pers.', 'corr: random pers.', 'error: random pers.'], loc=4)
    #plt.title('twitter network')
    plt.show()
    exit()
