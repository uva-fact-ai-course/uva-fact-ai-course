import numpy as np
import pandas as pd


def objective_heuristic(weights, rankings):
    n = len(weights)

    Ais = pd.Series(np.zeros(n))
    Ris = pd.Series(np.zeros(n))

    for ranking in rankings:
        assert len(ranking) == n

        priority = Ais - Ris - ranking
        priority.sort_values(inplace=True, ascending=True)

        new_Ais = weights.copy()
        new_Ais.index = priority.index
        Ais += new_Ais

        Ris += ranking

        unf = (Ais - Ris).abs().sum()
        yield unf
