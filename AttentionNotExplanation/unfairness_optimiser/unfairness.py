import numpy as np
import pandas as pd


def unfairness_progression(weights, rankings):
    """Generates unfairness per ranking and iteration

    Args:
        weights (array-like): weighted attention per position
        rankings (array-like): stores single rankings

    Yields:
        float : unfairness score

    """
    n = len(weights)
    Ais = pd.Series(np.zeros(n))
    Ris = pd.Series(np.zeros(n))

    for ranking in rankings:
        assert len(ranking) == n

        new_Ais = weights.copy()
        new_Ais.index = ranking.index
        Ais += new_Ais

        Ris += ranking

        unf = (Ais - Ris).abs().sum()
        yield unf


def unfairness(weights, *rankings):
    """Calculates the unfairness for the entire rankings set

    Args:
        weights (array-like): weighted attention per position
        rankings (array-like): stores single rankings

    Returns:
        float : unfairness score of last result

    """
    result = None

    for result in unfairness_progression(weights, rankings):
        pass

    return result
