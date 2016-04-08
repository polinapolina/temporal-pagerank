__author__ = 'Polina'
import scipy.stats
import numpy as np
import operator

def get_binned_values(ordered_pr_ref, ordered_pr_out, bins_num):
    bins = -np.log(np.linspace(min(ordered_pr_ref + ordered_pr_out), max(ordered_pr_ref + ordered_pr_out), num = bins_num))
    ranking_ref = np.digitize(-np.log(ordered_pr_ref), bins)
    ranking_out = np.digitize(-np.log(ordered_pr_out), bins)
    return ranking_ref, ranking_out


def get_ordered(pr_ref, pr_out):
    rank_order = [key for (key, value) in sorted(pr_ref.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_pr_ref = np.array([pr_ref[k] for k in rank_order])
    ordered_pr_out = np.array([pr_out[k]/sum(pr_out.values()) for k in rank_order])
    return ordered_pr_ref, ordered_pr_out

def get_ordered_ranks(pr_ref, pr_out):
    rank_order = [key for (key, value) in sorted(pr_ref.items(), key=operator.itemgetter(1), reverse=True)]
    ordered_pr_ref = np.array([1.0 - pr_ref[k] for k in rank_order])
    ordered_pr_out = np.array([1.0 - pr_out[k]/sum(pr_out.values()) for k in rank_order])
    #ordered_pr_out__ = np.array([pr_out[k]/sum(pr_out.values()) for k in rank_order])
    out_rank = scipy.stats.rankdata(ordered_pr_out)
    ref_rank = scipy.stats.rankdata(ordered_pr_ref)
    return ref_rank, out_rank

def get_topk_corr(pr_ref, pr_out, k_range, bins = 0):

    ordered_pr_ref, ordered_pr_out = get_ordered(pr_ref, pr_out)

    if bins > 0:
        ref_rank, out_rank = get_binned_values(ordered_pr_ref, ordered_pr_out, bins)
    else:
        ref_rank = scipy.stats.rankdata(ordered_pr_ref)
        out_rank = scipy.stats.rankdata(ordered_pr_out)

    spearman_top = []
    tau_top = []
    for k in k_range:
        tau_top.append(scipy.stats.kendalltau(out_rank[:k], ref_rank[:k])[0])
        spearman_top.append(scipy.stats.spearmanr(out_rank[:k], ref_rank[:k])[0])
    return tau_top, spearman_top

def get_topk_corr_union(pr_ref, pr_out, k_range, bins = 0):

    pr_out = {k:v/sum(pr_out.values()) for (k,v) in pr_out.iteritems()}
    sorted_ref = sorted(pr_ref.items(), key=operator.itemgetter(1), reverse=True)
    sorted_out = sorted(pr_out.items(), key=operator.itemgetter(1), reverse=True)

    if bins > 0:
        ref_rank, out_rank = get_binned_values([v for (k, v) in sorted_ref], [v for (k, v) in sorted_out], bins)
    else:
        ref_rank = scipy.stats.rankdata([v for (k, v) in sorted_ref])
        out_rank = scipy.stats.rankdata([v for (k, v) in sorted_out])

    ref_rank_dict = {sorted_ref[i][0]: ref_rank[i] for i in xrange(len(ref_rank))}
    out_rank_dict = {sorted_out[i][0]: out_rank[i] for i in xrange(len(out_rank))}

    spearman_top = []
    tau_top = []

    for k in k_range:
        top_elements = set([i[0] for i in sorted_ref[:k]]+[i[0] for i in sorted_out[:k]])
        top_ref = [ref_rank_dict[i] for i in top_elements]
        top_out = [out_rank_dict[i] for i in top_elements]
        tau_top.append(scipy.stats.kendalltau(top_ref, top_out)[0])
        spearman_top.append(scipy.stats.spearmanr(top_ref, top_out)[0])
    return tau_top, spearman_top