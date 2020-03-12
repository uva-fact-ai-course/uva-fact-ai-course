"""
Example script for comparing occupational bias across word embeddings.
Following approach from:

Man is to Computer Programmer as Woman is to Homemaker?
    Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""

from __future__ import print_function, division

import argparse
import sys

from matplotlib import pyplot as plt

from debiaswe.data import load_professions, load_definitional_pairs
from debiaswe.we import doPCA

if sys.version_info[0] < 3:
    import io

    open = io.open


def plot_comparison_embeddings(
    datapoints_a, datapoints_b, embedding_names, name="compare_bias", save=False,
):
    """
    Plot the occupational bias comparison across two embeddings.


    :param list datapoints_a: datapoints of first embedding
    :param list datapoints_b: datapoints of second embedding
    :param list embedding_names: List with strings of names of
        embeddings. For example, ["word2vec", "GloVe"].
    :param string name: string with name of file save
    :param boolean save: whether to save plot
    :returns: None
    """
    fig, ax = plt.subplots()
    ax.scatter(datapoints_b, datapoints_a, s=10)
    ax.set_ylim(min(datapoints_a) - 0.1, max(datapoints_a) + 0.1)
    ax.set_xlim(min(datapoints_b) - 0.1, max(datapoints_b) + 0.1)
    plt.xlabel("Gender axis {}".format(embedding_names[1]), fontsize=12)
    plt.ylabel("Gender axis {}".format(embedding_names[0]), fontsize=12)
    plt.title("Gender bias in professions across embeddings", pad=18, fontsize=13)
    if save:
        fig.savefig("{}.png".format(name))
    plt.show()


def get_datapoints_embedding(E, v_gender, professions, unique_occupations):
    """
    Get datapoints for one embedding.


    :param object E: WordEmbedding object.
    :param list projection: List with projection of profession words
        onto gender axis of embedding.
    :param list unique_occupations: List of occupations present in all
        embeddings to compare.
    :returns: datapoints list
    """
    # Extract datapoint per occupation and sort datapoints
    sp = sorted(
        [(E.v(w).dot(v_gender), w) for w in professions if w in unique_occupations]
    )
    points = [s[0] for s in sp]
    words = [s[1] for s in sp]
    words_sorted_ind = sorted(range(len(words)), key=lambda k: words[k])
    datapoints = [points[i] for i in words_sorted_ind]
    return datapoints


def project_profession_words(E, professions):
    """
    Get gender axis and project profession words onto this axis.

    :param object E: WordEmbedding object.
    :param list professions: List of professions
    :param list unique_words: List of words present in all
        embeddings to compare.
    :returns: projection, profession words, gender axis
    """
    # Extract definitional word embeddings and determine gender direction.
    defs = load_definitional_pairs(E.words)

    v_gender = doPCA(defs, E).components_[0]

    # Projection on the gender direction.
    sp = E.profession_stereotypes(professions, v_gender, print_firstn=0)

    occupations = [s[1] for s in sp]
    return sp, occupations, v_gender


def compare_occupational_bias(
    E_a, E_b, embedding_names, name="compare_bias", save=True
):
    """
    Compare occupational bias across word embeddings.

    :param object E_a: WordEmbedding object.
    :param object E_b: WordEmbedding object.
    :param list embedding_names: List with strings of names of
        embeddings. For example, ["word2vec", "GloVe"].
    :param string name: string with name of file save
    :param save: 
    :returns: None
    """
    professions = load_professions()
    proj_a, prof_a, v_gender_a = project_profession_words(E_a, professions)
    proj_b, prof_b, v_gender_b = project_profession_words(E_b, professions)
    unique_occupations = list(set(prof_a).intersection(prof_b))
    datapoints_a = get_datapoints_embedding(
        E_a, v_gender_a, professions, unique_occupations
    )
    datapoints_b = get_datapoints_embedding(
        E_b, v_gender_b, professions, unique_occupations
    )
    plot_comparison_embeddings(
        datapoints_a, datapoints_b, embedding_names, name, save=save
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename_a", help="The name of the embedding")
    parser.add_argument(
        "embedding_filename_b", help="The name of the embedding to compare with"
    )
    parser.add_argument(
        "embedding_names", type=list, help="List of two strings with embedding names"
    )
    parser.add_argument(
        "--save", type=bool, default=False, help="If true plot is saved"
    )
    parser.add_argument(
        "--name", type=str, default="compare_bias", help="Name of plot save"
    )

    if len(sys.argv[2] != 2):
        print("Please give third argument the names of two embeddings as list")

    args = parser.parse_args()

    compare_occupational_bias(
        args.embedding_filename_a,
        args.embedding_filename_b,
        args.embedding_names,
        args.name,
    )
