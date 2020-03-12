"""
Tools for benchmarking word embeddings.

Code adapted and extended from:
https://github.com/k-kawakami/embedding-evaluation
and
https://github.com/chadaeun/weat_replication

Using well-known benchmarks from:
(MSR)
    T. Mikolov, W.-t. Yih, and G. Zweig.
    Linguistic regularities in continuous space word representations.
    2013.
(RG)
    H. Rubenstein and J. B. Goodenough.
    Contextual correlates of synonymy.
    1965.
(WS)
    L. Finkelstein, E. Gabrilovich, Y. Matias, E. Rivlin, Z. Solan,
        G. Wolfman, and E. Ruppin.
    Placing search in context: The concept revisited.
    2001.
(WEAT)
    Aylin Caliskan, Joanna J Bryson, and ArvindNarayanan.
    Semantics derived automatically from language corpora contain
        human-like biases.
     2017.
"""

import os
import time
from collections import defaultdict

import numpy as np
from scipy import linalg, stats
from tqdm import tqdm

from .data import load_professions, load_definitional_pairs
from .we import doPCA

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


class Benchmark:
    def __init__(self):
        self.DATA_ROOT = os.path.join(PKG_DIR, "benchmark_data")
        self.v_gender = None
        self.files = [
            file_name.replace(".txt", "")
            for file_name in os.listdir(self.DATA_ROOT)
            if ".txt" in file_name
        ]
        self.dataset = defaultdict(list)
        for file_name in self.files:
            for line in open(self.DATA_ROOT + "/" + file_name + ".txt"):
                self.dataset[file_name].append(
                    [
                        float(w) if i == 2 else w
                        for i, w in enumerate(line.strip().split())
                    ]
                )

    @staticmethod
    def cos(vec1, vec2):
        """
        Calculates cosine similarity between two NumPy vectors


        :param ndarray vec1: NumPy vector 1
        :param ndarray vec2: NumPy vector 2
        :returns: cosine similarity
        """
        return vec1.dot(vec2) / (linalg.norm(vec1) * linalg.norm(vec2))

    @staticmethod
    def rho(vec1, vec2):
        """
        Calculates Spearman Rho between two NumPy vectors


        :param ndarray vec1: NumPy vector 1
        :param ndarray vec2: NumPy vector 2
        :returns: Rho
        """
        return stats.stats.spearmanr(vec1, vec2)[0]

    @staticmethod
    def pprint(result, title):
        """
        Plots benchmark results


        :param dictionary result: Dictionary with benchmark names as keys and
            lists containing [number of found words, number of missing words,
            benchmark result] as values.
        :param string title: Title of the table.
        :returns: None
        """
        from prettytable import PrettyTable

        table = PrettyTable(["Dataset", "Found", "Not Found", "Score"])
        table.title = "Results for {}".format(title)
        table.align["Dataset"] = "l"
        for k, v in result.items():
            table.add_row([k, v[0], v[1], v[2]])
        print(table)

    @staticmethod
    def pprint_compare(results, methods, title):
        """
        Plots benchmark results for all methods in one table to compare.


        :param list result: List of dictionaries with benchmark names as keys
            and lists containing [number of found words, number of missing
            words, benchmark result] as values.
        :param list methods: List of method names.
        :param string title: Title of the table.
        :returns: None
        """
        assert len(results) == len(methods)
        from prettytable import PrettyTable

        table = PrettyTable(["Score", "RG-65", "WS-353", "MSR", "WEAT"])
        table.title = "Results for {}".format(title)
        for result, method in zip(results, methods):
            table.add_row(
                [
                    method,
                    list(result["EN-RG-65"])[2],
                    list(result["EN-WS-353-ALL"])[2],
                    list(result["MSR-analogy"])[2],
                    list(result["WEAT"])[2],
                ]
            )
        print(table)

    def evaluate(
        self, E, title, discount_query_words=False, batch_size=200, print=True
    ):
        """
        Evaluates RG-65, WS-353, MSR, and WEAT benchmarks


        :param object E: WordEmbedding object.
        :param string title: Title of the results table.
        :param int batch_size: Size of the batches in which to process
            the queries.
        :param boolean discount_query_words: Give analogy solutions that appear
            in the query 0 score in MSR benchmark. (Default = False)
        :param boolean print: Print table with results. (Default = True)
        :returns: dict with results
        """
        word_dict = E.get_dict()
        result = {}
        vocab = word_dict.keys()
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [], [], 0, 0
            for datum in data:
                if datum[0] in vocab and datum[1] in vocab:
                    found += 1
                    pred.append(self.cos(word_dict[datum[0]], word_dict[datum[1]]))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = [found, notfound, self.rho(label, pred) * 100]
        msr_res = self.MSR(E, discount_query_words, batch_size)
        result["MSR-analogy"] = [msr_res[1], msr_res[2], msr_res[0]]
        weat_res = self.weat(E)
        result["WEAT"] = ["-", "-", weat_res]
        if print:
            self.pprint(result, title)
            time.sleep(3)

        return result

    def MSR(self, E, discount_query_words=False, batch_size=200):
        """
        Executes MSR-analogy benchmark on the word embeddings in E


        :param WordEmbedding E: WordEmbedding object containing embeddings.
        :param boolean discount_query_words: Give analogy solutions that appear
            in the query 0 score. (Default = False)
        :param int batch_size: Size of the batches in which to process
            the queries. (Default = 200)
        :returns: Percentage of correct analogies (accuracy),
            number of queries without OOV words,
            number of queries with OOV words
        """
        # Load and format the benchmark data
        analogy_answers = np.genfromtxt(
            self.DATA_ROOT + "/word_relationship.answers", dtype="str", encoding="utf-8"
        )
        analogy_a = np.expand_dims(analogy_answers[:, 1], axis=1)
        analogy_q = np.genfromtxt(
            self.DATA_ROOT + "/word_relationship.questions",
            dtype="str",
            encoding="utf-8",
        )

        # Remove Out Of Vocabulary words_not_found
        analogy_stack = np.hstack((analogy_a, analogy_q))
        present_words = np.isin(analogy_stack, E.words).all(axis=1)
        filtered_answers = analogy_a[present_words]
        filtered_questions = analogy_q[present_words]

        # Batch the queries up
        y = []
        n_batches = len(analogy_answers) // batch_size
        for i, batch in enumerate(tqdm(np.array_split(filtered_questions, n_batches))):
            # Extract relevant embeddings from E
            a = E.vecs[np.vectorize(E.index.__getitem__)(batch[:, 0])]
            x = E.vecs[np.vectorize(E.index.__getitem__)(batch[:, 1])]
            b = E.vecs[np.vectorize(E.index.__getitem__)(batch[:, 2])]
            all_y = E.vecs

            # Calculate scores
            batch_pos = ((1 + all_y @ x.T) / 2) * ((1 + all_y @ b.T) / 2)
            batch_neg = (1 + all_y @ a.T + 0.00000001) / 2
            batch_scores = batch_pos / batch_neg

            # If set, set scores of query words to 0
            if discount_query_words:
                query_ind = np.vectorize(E.index.__getitem__)(batch).T
                batch_scores[query_ind, np.arange(batch_scores.shape[1])[None, :]] = 0

            # Retrieve words with best analogy scores
            y.append(np.array(E.words)[np.argmax(batch_scores, axis=0)])

        # Calculate returnable metrics
        y = np.hstack(y)[:, None]
        accuracy = np.mean(y == filtered_answers) * 100
        words_not_found = len(analogy_answers) - len(filtered_answers)

        return accuracy, len(filtered_answers), words_not_found

    def weat(self, E):
        """
        Calculated WEAT effect size of association between female and male
        words as attributes and typically female and male professions as
        target words. Score is [-2,2], where closer to 0 is less biased.


        :param WordEmbedding E: WordEmbedding object containing embeddings.
        :returns: effect size
        """
        # Extract definitional word embeddings and determine gender direction.
        defs = load_definitional_pairs(E.words)
        unzipped_defs = list(zip(*defs))
        female_defs = np.array(unzipped_defs[0])
        male_defs = np.array(unzipped_defs[1])
        A = E.vecs[np.vectorize(E.index.__getitem__)(female_defs)]
        B = E.vecs[np.vectorize(E.index.__getitem__)(male_defs)]
        # Determine gender direction if nescessary
        v_gender = None
        if self.v_gender is None:
            v_gender = doPCA(defs, E).components_[0]
            self.v_gender = v_gender
        else:
            v_gender = self.v_gender

        # Extract professions and split according to projection on the gender
        # direction.
        professions = load_professions(embed_words=E.words)
        sp = sorted([(E.v(w).dot(v_gender), w) for w in professions])
        unzipped_sp = list(zip(*sp))
        prof_scores = np.array(unzipped_sp[0])
        sorted_profs = np.array(unzipped_sp[1])
        female_prof = sorted_profs[prof_scores > 0]
        male_prof = sorted_profs[prof_scores < 0]

        # Balance target sets and extract their embeddings.
        female_prof, male_prof = self.balance_word_vectors(female_prof, male_prof)

        X = E.vecs[np.vectorize(E.index.__getitem__)(np.array(female_prof))]
        Y = E.vecs[np.vectorize(E.index.__getitem__)(np.array(male_prof))]

        # Calculate effect size
        x_assoc = np.mean((X @ A.T), axis=-1) - np.mean((X @ B.T), axis=-1)
        y_assoc = np.mean((Y @ A.T), axis=-1) - np.mean((Y @ B.T), axis=-1)

        num = np.mean(x_assoc, axis=-1) - np.mean(y_assoc, axis=-1)
        denom = np.std(np.concatenate((x_assoc, y_assoc), axis=0))

        return num / denom

    @staticmethod
    def balance_word_vectors(A, B):
        """
        Balance size of two lists of word vectors by randomly deleting some
        vectors in the larger one.


        :param ndarray A: numpy ndarrary of word vectors
        :param ndarray B: numpy ndarrary of word vectors
        :return: tuple of two balanced word vector matrixes
        """

        diff = len(A) - len(B)

        if diff > 0:
            A = np.delete(A, np.random.choice(len(A), diff, 0), axis=0)
        else:
            B = np.delete(B, np.random.choice(len(B), -diff, 0), axis=0)

        return A, B
