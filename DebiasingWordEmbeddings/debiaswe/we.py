"""
Tools for debiasing word embeddings.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

import os

import numpy as np
import scipy.sparse
from sklearn.decomposition import PCA

from .download import download
from .embeddings_config import ID


class WordEmbedding:
    def __init__(self, embedding, limit=None):
        self._words = []
        self._vecs = None
        self._index = None
        self.thresh = None
        self.max_words = None
        self.desc = embedding

        from_file = False
        fname = None

        # If embedding in standard available embeddings, check if download
        # is needed.
        if embedding in ID:
            extension = ID[embedding]["extension"]
            fname = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "embeddings",
                embedding + extension,
            )
            # If embedding file not present, download it
            if not os.path.exists(fname):
                download(embedding)
        else:
            # Check for file path
            assert os.path.exists(embedding) and os.path.isfile(
                embedding
            ), "Not an available embedding or known file path"
            # If valid file available, load from that file
            from_file = True
            fname = embedding
            print(f"Creating embedding from: {os.path.abspath(embedding)}")

        # Load large embeddings using gensim
        if embedding.endswith("large") or (
            len(embedding.split(".")) > 1 and embedding.split(".")[-2].endswith("large")
        ):
            import gensim.models

            model = gensim.models.KeyedVectors.load_word2vec_format(
                fname, binary=fname.endswith(".bin"), limit=limit
            )
            self._words = sorted(
                [w for w in model.vocab], key=lambda w: model.vocab[w].index
            )
            self._vecs = np.array([model[w] for w in self._words], dtype="float32")
        # Load small files by reading line by line
        else:
            vecs = []
            words = []
            # Open and read from file
            with open(fname, "r", encoding="utf8") as f:
                for line in f:
                    if len(words) == limit:
                        break
                    s = line.split()
                    v = np.array([float(x) for x in s[1:]])
                    words.append(s[0])
                    vecs.append(v)
            # Determine correct (i.e. most common) vector length
            lengths = [len(v) for v in vecs]
            correct_length = max(lengths, key=lengths.count)
            # Filter out any loaded vectors with uncommon length
            vecs_filtered = []
            for v in vecs:
                if len(v) == correct_length:
                    vecs_filtered.append(v)
                elif from_file:
                    print("Got weird line:", line)
            self._vecs = np.array(vecs_filtered, dtype="float32")
            self._words = words

        # Reindex and if needed normalize after loading
        self.reindex()
        norms = np.linalg.norm(self._vecs, axis=1)
        if max(norms) - min(norms) > 0.0001:
            if from_file:
                print("Normalizing vectors...")
            self.normalize()
        print("Embedding shape:", self._vecs.shape)
        print(
            self.n,
            "words of dimension",
            self.d,
            ":",
            ", ".join(self._words[:4] + ["..."] + self._words[-4:]),
        )

    def get_dict(self):
        """
        Converts the saved words and their embeddings to dictionary format.


        :returns: Dictionary with words as keys and embeddings as values
        """
        return {key: value for key, value in zip(self._words, self._vecs)}

    @property
    def words(self):
        """
        Retrieve the words from the embeddings


        :returns: List of words
        """
        return self._words

    @words.setter
    def words(self, words):
        """
        Set the words of the embeddings.


        :param list words: List of words.
        """
        self._words = words

    @property
    def vecs(self):
        """
        Retrieve the embeddings


        :returns: ndarray of embeddings
        """
        return self._vecs

    @vecs.setter
    def vecs(self, vecs):
        """
        Set the embeddings.


        :param ndarray vecs: NumPy array of embeddings
        """
        self._vecs = vecs

    @property
    def index(self):
        """
        Retrieve the indices of the words' embeddings


        :returns: Dictionary with words as keys and the indices of their
            embeddings as values
        """
        return self._index

    @index.setter
    def index(self, index):
        """
        Set the indices.


        :param dictionary index: Dictionary of words:indices.
        """
        self._index = index

    def reindex(self):
        """
        Generate dictionary of indices and check consistency in size.
        """
        self._index = {w: i for i, w in enumerate(self._words)}
        self.n, self.d = self._vecs.shape
        assert self.n == len(self._words) == len(self._index)
        self._neighbors = None

    def v(self, word):
        """
        Retrieve the embedding vector of a word.


        :param string word: Word to retrieve the embedding of.
        :returns: ndarray with the embedding
        """
        return self._vecs[self._index[word]]

    def diff(self, word1, word2):
        """
        Calculate the normalized distance between the embeddings of two words.


        :param string word1: First word of which to compare the embedding.
        :param string word2: Second word of which to compare the embedding.
        :returns: Normalized difference vector
        """
        v = self._vecs[self._index[word1]] - self._vecs[self._index[word2]]
        return v / np.linalg.norm(v)

    def normalize(self):
        """
        Normalize all embedding vectors.
        """
        self.desc += ", normalize"
        self._vecs /= np.linalg.norm(self._vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def shrink(self, numwords):
        """
        Reduce the number of words and embeddings.


        :param integer numwords: number of words and embeddings to retain.
        """
        self.desc += ", shrink " + str(numwords)
        self.filter_words(lambda w: self._index[w] < numwords)

    def filter_words(self, test):
        """
        Keep some words based on test, e.g. lambda x: x.lower()==x.


        :param lambda test: Lambda function detailing condition to keep a word.
        """
        self.desc += ", filter"
        kept_indices, words = zip(
            *[[i, w] for i, w in enumerate(self._words) if test(w)]
        )
        self._words = list(words)
        self._vecs = self._vecs[kept_indices, :]
        self.reindex()

    def save(self, filename):
        """
        Save the words and embeddings to a file.


        :param string filename: Location to save the embeddings to.
        """
        with open(filename, "w", encoding="utf8") as f:
            f.write(
                "\n".join(
                    [
                        w + " " + " ".join([str(x) for x in v])
                        for w, v in zip(self._words, self._vecs)
                    ]
                )
            )
        print("Wrote", self.n, "words to", filename)

    def save_embeddings(self, filename, binary=True):
        """
        Save the words and embeddings to a file, sorted by words frequency
            in descending order.


        :param string filename: Location to save the embeddings to.
        :param boolean binary: Save as a binary file. (Default = True)
        """
        with open(filename, "wb", encoding="utf8") as fout:
            fout.write("%s %s\n" % self._vecs.shape)
            # store in sorted order: most frequent words at the top
            for i, word in enumerate(self._words):
                row = self._vecs[i]
                if binary:
                    fout.write(word + b" " + row.tostring())
                else:
                    fout.write(
                        "%s %s\n" % (word, " ".join("%f" % val for val in row))
                    )

    def compute_neighbors_if_necessary(self, thresh, max_words):
        """
        Calculate which embeddings are close to each embedding.


        :param float thresh: Threshold to determine when an embedding is close.
        :param integer max_words: Maximum number of words to considder.
        """
        thresh = float(thresh)  # dang python 2.7!
        if (
            self._neighbors is not None
            and self.thresh == thresh
            and self.max_words == max_words
        ):
            return
        print("Computing neighbors...")
        self.thresh = thresh
        self.max_words = max_words
        vecs = self._vecs[:max_words]
        dots = vecs.dot(vecs.T)
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1 - thresh / 2))
        from collections import Counter

        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean number of neighbors per word:", np.mean(nums) - 1)
        print("Median of number of neighbors per word:", np.median(nums) - 1)
        rows, cols, vecs = zip(
            *[
                (i, j, vecs[i] - vecs[j])
                for i, j, x in zip(rows, cols, dots.data)
                if i < j
            ]
        )
        self._neighbors = rows, cols, np.array([v / np.linalg.norm(v) for v in vecs])

    def best_analogies_dist_thresh(self, v, thresh=1, topn=500, max_words=50000):
        """
        Generate analogies based on the seed direction using the metric
            'cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0'
            to select neighbors.


        :param ndarray v: Seed direction along which the analogies are
            generated
        :param float thresh: Threshold to determine when an embedding is close.
            (Default = 1)
        :param integer topn: Number of analogies to print.
        :param integer max_words: Maximum number of words to considder.
        :returns: List of analogies
        """
        vecs, vocab = self._vecs[:max_words], self._words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, vecs = self._neighbors
        scores = vecs.dot(v / np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL = set()
        usedR = set()
        for i in pi:
            if abs(scores[i]) < 0.001:
                break
            row = rows[i] if scores[i] > 0 else cols[i]
            col = cols[i] if scores[i] > 0 else rows[i]
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans) == topn:
                break

        return ans

    def profession_stereotypes(self, profession_words, bias_space, print_firstn=20):
        """
        Print the most stereotypical professions on both ends of the bias
            direction.


        :param list profession_words: List of profession words to project.
        :param ndarray bias_space: Direction on which the professions are
            projected.
        :param integer print_firstn: Number of professions to print.
        :returns: Sorted list of projected professions
        """
        assert isinstance(print_firstn, int) and print_firstn >= 0
        # Calculate the projection values onto the bias subspace
        sp = sorted(
            [
                (self.v(w).dot(bias_space), w)
                for w in profession_words
                if w in self._words
            ]
        )
        # Check what genders belong to positive/negative projection values
        pos_neg = (
            ("Female", "Male")
            if self.v("she").dot(bias_space) > 0
            else ("Male", "Female")
        )
        # Print the professions with scores
        if print_firstn > 0:
            print(pos_neg[0].center(38) + "|" + pos_neg[1].center(38))
            print("-" * 77)
        for i in range(min(print_firstn, len(sp))):
            print(
                str(sp[-(i + 1)][0].round(3)).ljust(8)  # score neg
                + sp[-(i + 1)][1].rjust(29)
                + " | "  # profession neg
                + sp[i][1].ljust(29)  # score pos
                + str(sp[i][0].round(3)).rjust(8)
            )  # profession pos
        return sp


def viz(analogies):
    """
    Print the analogies in a nicer format.


    :param list analogies: List of analogies to print.
    :returns: None
    """
    print("Index".ljust(12) + "Analogy".center(45) + "Gender score".rjust(12))
    print("-" * 69)
    print(
        "\n".join(
            str(i).rjust(4) + a[0].rjust(29) + " | " + a[1].ljust(29) + (str(a[2]))[:4]
            for i, a in enumerate(analogies)
        )
    )


def doPCA(pairs, embedding, num_components=10):
    """
    Perform PCA on the centered embeddings of the words in the pairs.


    :param list pairs: List of word pairs defining the gender space.
    :param WordEmbedding embedding: WordEmbedding object containing embeddings.
    :param integer num_components: Number of principal components.
        (Default = 10)
    :returns: pca object with principal components.
    """
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b)) / 2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components=num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca


def drop(u, v):
    """
    Drop the bias subspace


    :param ndarray u: NumPy array with an embedding from which to remove the
        bias subspace.
    :param ndarray v: NumPy array with bias subspace.
    :returns: Debiased embeddings
    """
    return u - v * (u.dot(v) / v.dot(v))
