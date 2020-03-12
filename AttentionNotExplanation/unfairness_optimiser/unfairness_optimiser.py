import math

import numpy as np
import pandas as pd
import pulp

from .dcg import calculate_idcg


def generate_weights(n, k, p):
    """Generates w_j based on geometric distribution

    Args:
        n (int): number of total subjects
        k (int): number of relevant subjects with non-zero weights
        p (float): parameter for geometric distribution

    Returns:
        array: Weight values based on geometric distribution for relevant
            subjects

    """
    result = np.concatenate([np.logspace(1, k, k, base=p),
                             np.zeros(n - k)])
    result /= result.sum()
    return result


class UnfairnessOptimizer:
    """Implements the ILP Solver

    """
    def __init__(self, t, k, p, theta, solver=None):
        """Initializes the Optimiser

        Args:
            t (int): number of subjects to prefilter
            k (int): selected top rank positions
            p (float): parameter for geometric distribution
                theta (float): value to scale idcg value for objective funtion
                solver (pulp.Solver): allows non-default optimisation software
        """
        self.Ai = None
        self.Ri = None
        self.t = t
        self.k = k
        self.p = p
        self.theta = theta
        self.w = generate_weights(self.t, self.k, self.p)

        self.solver = solver

        assert self.k <= self.t

    @property
    def n(self):
        return self.Ai.shape[0]

    def _add_constraint_over_subjects(self, problem, pX):
        """Adds constraints for the subjects

        Args:
            problem (pulp.Problem): The generated pulp optimisation problem
            pX (dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
        """
        prefilter_indices = {pos[0] for pos in pX.keys()}
        for i in prefilter_indices:
            problem += (
                pulp.lpSum([pX[pos] for pos in pX.keys() if pos[0] == i]) == 1)

    def _add_constraint_over_positions(self, problem, pX):
        """Adds constraints for the positions

        Args:
            problem (pulp.Problem): The generated pulp optimisation problem
            pX (dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
        """
        for j in range(0, self.t):
            problem += (
                pulp.lpSum([pX[pos] for pos in pX.keys() if pos[1] == j]) == 1)

    def _add_quality_constraint(self, problem, pX, r, idcg):
        """Adds the quality constraint

        Args:
            problem (pulp.Problem): The generated pulp optimisation problem
            pX (dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
            r (array-like): relevance scores
            idcg (float): idcg score of ranking

        """
        prefilter_indices = {pos[0] for pos in pX.keys()}

        dcg_constraint = []
        for j in range(0, self.k):
            for i in prefilter_indices:
                v = 2 ** r[i] - 1
                v /= math.log2(j + 2)
                v *= pX[(i, j)]

                dcg_constraint.append(v)

        problem += (pulp.lpSum(dcg_constraint) >= self.theta * idcg)

    def _add_unfairness_objective(self, problem, pX, r):
        """Adds the optimization objective

        Args:
            problem (pulp.Problem): The generated pulp optimisation problem
            pX (dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
            r (array-like): relevance scores

        """
        prefilter_indices = {pos[0] for pos in pX.keys()}

        unfairness_objective = []
        for i in prefilter_indices:
            for j in range(0, self.t):
                v = abs((self.Ai[i] + self.w[j]) - (self.Ri[i] + r[i]))
                v *= pX[(i, j)]
                unfairness_objective.append(v)

        problem += pulp.lpSum(unfairness_objective)

    def _initialize_state(self, relevances):
        """Attention and Relevance values are initialized with zeros

        Args:
            relevances (array_like): relevance scores
        """
        n = len(relevances)
        if self.Ai is None:
            assert self.t <= n

            self.Ai = np.zeros(n)
            self.Ri = np.zeros(n)
        else:
            assert self.n == n

    def _update_state(self, pX, r):
        """Updates state after iteration

        Args:
            pX (dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
            r (array-like): relevance scores
        """
        for pos, variable in pX.items():
            if pulp.value(variable) == 1:
                self.Ai[pos[0]] += self.w[pos[1]]

        self.Ri += r

    def _initialize_LP(self, prefilter_indices):
        """Initializes the LP Problem in PulP

        Args:
            prefilter_indices (array-like): indices of prefilterd candidates
        """
        prob = pulp.LpProblem("Attention", pulp.LpMinimize)
        pX = pulp.LpVariable.dicts(
            'X',
            ((i, j) for i in prefilter_indices for j in range(0, self.t)),
            cat=pulp.LpBinary)

        return prob, pX

    def _prefilter(self, relevances):
        """Prefilters the relevance scores

        Args:
            relevances (array-like): relevance scores

        """
        position_to_relevance = pd.Series(relevances)

        top_k = position_to_relevance.sort_values(ascending=False).head(self.k)

        negative_promotion_worthiness = pd.Series(
            self.Ai - (self.Ri + relevances))
        negative_promotion_worthiness = \
            negative_promotion_worthiness.sort_values(ascending=True)

        worthiness_addition = negative_promotion_worthiness[
            ~negative_promotion_worthiness.index.isin(top_k.index)]
        worthiness_addition = worthiness_addition.head(self.t - self.k)

        prefilter_candidates = pd.concat([top_k, worthiness_addition])
        prefilter_candidates = position_to_relevance[
            prefilter_candidates.index]

        return prefilter_candidates

    def _rerank(self, pX, relevances):
        """Rekanks subjects after relevance scores

        Args:
            px dictionary): PuLP dictionary that stores assigned values
                for subject and ranking
            relevances (array-like): relevance scores

        """
        position_to_relevance = pd.Series(relevances)
        prefilter_indices = {pos[0] for pos in pX.keys()}

        ranking_positions = np.zeros(self.n, dtype=int)

        for pos, variable in pX.items():
            if pulp.value(variable) == 1:
                ranking_positions[pos[0]] = pos[1]

        # maybe order by relevance?
        filler_indices = position_to_relevance[
            ~position_to_relevance.index.isin(prefilter_indices)].index

        for i, new_pos in zip(filler_indices, range(self.t, self.n)):
            ranking_positions[i] = new_pos

        return ranking_positions

    def _optimise_unfairness_get_X(self, relevances):
        """Generates and solves the Linear Problem

        Args:
            relevances (array-like): relevance scores
        """
        self._initialize_state(relevances)

        prefilter_candidates = self._prefilter(relevances)
        prefilter_indices = list(prefilter_candidates.index)

        prob, pX = self._initialize_LP(prefilter_indices)

        self._add_constraint_over_subjects(prob, pX)
        self._add_constraint_over_positions(prob, pX)

        idcg = calculate_idcg(prefilter_candidates.to_numpy(), self.k)
        self._add_quality_constraint(prob, pX, relevances, idcg)

        self._add_unfairness_objective(prob, pX, relevances)

        status = prob.solve(solver=self.solver)
        assert status == pulp.LpStatusOptimal, pulp.LpStatus[status]

        self._update_state(pX, relevances)

        return pX

    def optimise_unfairness(self, relevances):
        """Optimises unfairness in the LP

        Args:
            relevances (array-like): relevance scores
        """
        self._optimise_unfairness_get_X(relevances)

        return np.absolute(self.Ai - self.Ri).sum()

    def optimise_unfairness_rerank(self, relevances):
        """Reranks subjects

        Args:
            relevances (array-like): relevance scores
        """
        pX = self._optimise_unfairness_get_X(relevances)

        return self._rerank(pX, relevances)
