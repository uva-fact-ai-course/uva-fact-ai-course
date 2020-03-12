import numpy as np
import pandas as pd
import pulp

from ..unfairness_optimiser import UnfairnessOptimizer
from ..unfairness_optimiser import generate_weights
from ..unfairness import unfairness_progression
from .heuristics import objective_heuristic


try:
    from tqdm.auto import trange
except Exception:
    trange = range


_solver = pulp.GUROBI(mip=True, msg=False)
if _solver.available() is False:
    _solver = None


def single_query(ranking, t, k, p, theta, iterations):
    global _solver

    optimiser = UnfairnessOptimizer(t, k, p, theta, solver=_solver)

    unfairness_track = []
    for i in trange(iterations):
        unfairness = optimiser.optimise_unfairness(ranking)
        unfairness_track.append(unfairness)

    return unfairness_track


def multi_query_interleave(rankings, iterations):
    for i in trange(iterations):
        yield from rankings


# rankings is a list of rankings per column
# so [ranking['score'], ranking['cleanliness'], etc]
# for the seven mentioned attributes
def multi_query(rankings, t, k, p, theta, iterations):
    global _solver

    optimiser = UnfairnessOptimizer(t, k, p, theta, solver=_solver)

    unfairness_track = []
    for ranking in multi_query_interleave(rankings, iterations):
        unfairness = optimiser.optimise_unfairness(ranking)
        unfairness_track.append(unfairness)

    return unfairness_track


def single_query_relevance(ranking, k, p, iterations):
    n = len(ranking)
    weights = pd.Series(generate_weights(n, k, p))
    ranking = pd.Series(np.sort(ranking)[::-1])

    rankings = (ranking for _ in range(0, iterations))
    return list(unfairness_progression(weights, rankings))


def single_query_objective(ranking, k, p, iterations):
    n = len(ranking)
    weights = pd.Series(generate_weights(n, k, p))
    ranking = pd.Series(np.sort(ranking)[::-1])

    rankings = (ranking for _ in range(0, iterations))
    return list(objective_heuristic(weights, rankings))


def multi_query_relevance(rankings, k, p, iterations):
    n = len(rankings[0])
    weights = pd.Series(generate_weights(n, k, p))
    rankings = [pd.Series(np.sort(ranking)[::-1]) for ranking in rankings]

    rankings = multi_query_interleave(rankings, iterations)
    return list(unfairness_progression(weights, rankings))


def multi_query_objective(rankings, k, p, iterations):
    n = len(rankings[0])
    weights = pd.Series(generate_weights(n, k, p))
    rankings = [pd.Series(np.sort(ranking)[::-1]) for ranking in rankings]

    rankings = multi_query_interleave(rankings, iterations)
    return list(objective_heuristic(weights, rankings))
