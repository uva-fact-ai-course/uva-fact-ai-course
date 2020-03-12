"""
Tools for data operations.
Extended from the code from:

Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

import json
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_professions(load_scores=False, embed_words=None):
    """
    Loads professions from data file


    :param bool load_scores: Load bias scores if True. Defaults to False.
    :param list embed_words: Available embedding words.
    :returns: List of professions
    """
    assert embed_words is None or type(embed_words) == list

    professions_file = os.path.join(PKG_DIR, "data", "professions.json")
    with open(professions_file, "r") as f:
        professions = json.load(f)

    # Embedding words + load scores: filter words
    if embed_words is not None and load_scores:
        professions = [p for p in professions if p[0] in embed_words]
    # Embedding words + not load scores: filter words and scores
    elif embed_words is not None and not load_scores:
        professions = [p[0] for p in professions if p[0] in embed_words]
    # No embedding words + not load scores: filter scores
    elif not load_scores:
        professions = [p[0] for p in professions if p[0]]
    # No embedding words + load scores: do nothing

    return professions


def load_gender_seed(embed_words=None):
    """
    Loads gender seed words from data file


    :param list embed_words: Available embedding words.
    :returns: List of gender-specific words
    """
    assert embed_words is None or type(embed_words) == list

    gender_file = os.path.join(PKG_DIR, "data", "gender_specific_seed.json")
    with open(gender_file, "r") as f:
        gender_words = json.load(f)

    # Filter unknown words if nescessary
    if embed_words is not None:
        gender_words = [w for w in gender_words if w in embed_words]

    return gender_words


def load_equalize_pairs():
    """
    Loads equalize pairs from data file


    :returns: List of equalize pairs
    """
    eq_file = os.path.join(PKG_DIR, "data", "equalize_pairs.json")
    with open(eq_file, "r") as f:
        eq_pairs = json.load(f)

    return eq_pairs


def load_definitional_pairs(embed_words=None):
    """
    Loads definitional pairs from data file


    :param list embed_words: Available embedding words.
    :returns: List of definitional pairs
    """
    assert embed_words is None or type(embed_words) == list

    def_file = os.path.join(PKG_DIR, "data", "definitional_pairs.json")
    with open(def_file, "r") as f:
        def_pairs = json.load(f)

    # Filter unknown words if necessary
    if embed_words is not None:
        def_pairs = [
            p for p in def_pairs if p[0] in embed_words and p[1] in embed_words
        ]

    return def_pairs


def load_data(embed_words=None):
    """
    Loads all data needed for debiasing and inspecting gender bias
    in proffesions.


    :param list embed_words: Available embedding words.
    :returns: List of gender-specific words, list of definitional pairs,
        list of equalize pairs, list of professions
    """
    assert embed_words is None or type(embed_words) == list

    profs = load_professions(embed_words=embed_words)
    gender_seed = load_gender_seed(embed_words=embed_words)
    eq_pairs = load_equalize_pairs()
    def_pairs = load_definitional_pairs(embed_words=embed_words)
    return gender_seed, def_pairs, eq_pairs, profs
