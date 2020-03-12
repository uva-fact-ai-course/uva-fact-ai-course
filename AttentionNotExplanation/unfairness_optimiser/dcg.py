import numpy as np


def calculate_idcg(relevances, k):
    """Calculates the IDCG value based on relevances

    Args:
        relevances (array-like): relevance scores of subjects
        k (int): particular rank position up to which to calculate idcg

    Returns:
        float: The IDCG value
    """
    sorted_relevances = np.sort(relevances)[::-1]
    return calculate_dcg(sorted_relevances, k)


def calculate_dcg(relevances, k):
    """Calculates the DCG value based on relevances

    Args:
        relevances (array-like): relevance scores of subjects
        k (int): particular rank position up to which to calculate idcg

    Returns:
        float: The DCG value
    """
    relevances = relevances[:k]
    return ((2**relevances-1)/np.log2(np.arange(k)+2)).sum()
