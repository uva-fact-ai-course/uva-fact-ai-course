""" TAKEN FROM:

https://ariepratama.github.io/Introduction-to-BLEU-in-python/ """

import numpy as np
from nltk import ngrams
from collections import Counter

def count_ngram(unigram, ngram=1):
    """
    Return
    -----
    counter: dict, containing ngram as key, and count as value
    """
    return Counter(ngrams(unigram, ngram))

def count_clip_ngram(translation_u, list_of_reference_u, ngram=1):
    """
    Return
    ----
    
    """
    res = dict()
    # retrieve hypothesis counts
    ct_translation_u = count_ngram(translation_u, ngram=ngram)
    
    # retrieve translation candidate counts
    for reference_u in list_of_reference_u:
        ct_reference_u = count_ngram(reference_u, ngram=ngram)
        for k in ct_reference_u:
            if k in res:
                res[k] = max(ct_reference_u[k], res[k])
            else:
                res[k] = ct_reference_u[k]
                

    return {
        k: min(ct_translation_u.get(k, 0), res.get(k, 0)) 
        for k in ct_translation_u
    }

def modified_precision(translation_u, list_of_reference_u, ngram=1):
    ct_clip = count_clip_ngram(translation_u, list_of_reference_u, ngram)
    ct = count_ngram(translation_u, ngram)
    
    return sum(ct_clip.values()) / float(max(sum(ct.values()), 1))

def closest_ref_length(translation_u, list_of_reference_u):
    """
    determine the closest reference length from translation length
    """
    len_trans = len(translation_u)
    closest_ref_idx = np.argmin([abs(len(x) - len_trans) for x in list_of_reference_u])
    return len(list_of_reference_u[closest_ref_idx])

def brevity_penalty(translation_u, list_of_reference_u):
    """
    """
    c = len(translation_u)
    r = closest_ref_length(translation_u, list_of_reference_u)
    
    if c > r:
        return 1
    else:
        return np.exp(1 - float(r)/c)

def bleu_score(translation_u, list_of_reference_u, W=[0.25 for x in range(4)]):
    bp = brevity_penalty(translation_u, list_of_reference_u)
    modified_precisions = [
        modified_precision(translation_u, list_of_reference_u, ngram=ngram)
        for ngram, _ in enumerate(W,start=1)
    ]
    score = np.sum([
        wn * np.log(modified_precisions[i]) if modified_precisions[i] != 0 else 0 for i, wn in enumerate(W)
    ])
    
    return bp * np.exp(score)