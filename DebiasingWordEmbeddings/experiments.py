import argparse
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_data
from debiaswe.debias import hard_debias, soft_debias
from debiaswe.benchmarks import Benchmark
from copy import deepcopy

# Parameters for doing soft debiasing from scratch
SOFT_PARAMS = {
    "word2vec": {
        "epochs": 2000,
        "lr": 0.01,
        "gamma": 0.1,
        "decrease_times": [1000, 1500, 1800],
    },
    "glove": {
        "epochs": 3500,
        "lr": 0.01,
        "gamma": 0.1,
        "decrease_times": [2300, 2800, 3000],
    },
    "fasttext": {"epochs": 7000, "lr": 0.01, "gamma": 0.1, "decrease_times": [5000]},
}


def print_details():
    """Prints parameter details about the script."""
    print("#" * 18 + "EXPERIMENT DETAILS".center(20) + "#" * 18)
    print("#" + "#".rjust(55))
    print("# " + f"Analogies and occupations: {str(not FLAGS.no_show)}".ljust(53) + "#")
    print(
        "# " + f"Do soft debiasing from scratch: {str(FLAGS.do_soft)}".ljust(53) + "#"
    )
    print("# " + f"Perform benchmarks: {str(not FLAGS.no_bench)}".ljust(53) + "#")
    print("#" + "#".rjust(55))
    print("# " + "Performing experiments for the following embeddings:".ljust(53) + "#")
    for embed in FLAGS.embeddings:
        print(f"#\t- {embed}".ljust(49) + "#")
    print("#" + "#".rjust(55))
    print("#" * 56)


def show_bias(E, v_gender, profession_words, info="", n=40):
    """
    Shows gender analogies and occupational gender biases.


    :param WordEmbedding E: WordEmbedding object.
    :param ndarray v_gender: Gender direction vector (Numpy array).
    :param list professions_words: List of professions.
    :param string info: Information about embedding.
    :param integer n: Number of gender analogies to show.
    """
    # Show 40 analogies and occupational gender bias
    a_gender = E.best_analogies_dist_thresh(v_gender, thresh=1, topn=n)
    print("\n" + "#" * 8 + f"GENDER ANALOGIES ({info})".center(40) + "#" * 8)
    we.viz(a_gender)
    print("\n" + "#" * 8 + f"OCCUPATIONAL GENDER BIAS ({info})".center(40) + "#" * 8)
    _ = E.profession_stereotypes(profession_words, v_gender)


def hard(E, gender_words, defs, equalize_pairs):
    """
    Hard debiasing of word embedding E.

    :param WordEmbedding E: Biased word embedding E.
    :param list gender_words: List of gender specific words.
    :param list defs: List of tuples with definitional pairs.
    :param list equalize_pairs: List of tuples with equalize pairs.
    :returns: Hard debiased WordEmbedding object.
    """
    print("\nHard debiasing...")
    E_hard = deepcopy(E)
    hard_debias(E_hard, gender_words, defs, equalize_pairs)
    return E_hard


def soft(E, embed, gender_words, defs):
    """
    Soft debiasing of word embedding E.

    :param WordEmbedding E: Biased word embedding E.
    :param string embed: Name of the embedding.
    :param list gender_words: List of gender specific words.
    :param list defs: List of tuples with definitional pairs.
    :returns: Soft debiased WordEmbedding object.
    """
    print("\nSoft debiasing...")
    # If do_soft is True, do soft debiasing from scratch
    if FLAGS.do_soft:
        params = SOFT_PARAMS[embed.split("_")[0]]
        E_soft = deepcopy(E)
        soft_debias(
            E_soft,
            gender_words,
            defs,
            epochs=params["epochs"],
            lr=params["lr"],
            gamma=params["gamma"],
            decrease_times=params["decrease_times"],
        )
    # If do_soft is False, load precomputed soft debiased embedding
    else:
        E_soft = WordEmbedding(embed + "_soft_debiased")
    return E_soft


def run_benchmark(E, E_hard, E_soft, embed):
    """
    Performs RG, WS, MSR and WEAT benchmarks.

    :param WordEmbedding E: Biased word embedding.
    :param WordEmbedding E_hard: Hard debiased word embedding.
    :param WordEmbedding E_soft: Soft debiased word embedding.
    :param string embed: Name of the embedding.
    """
    print("\nRunning benchmarks...")
    benchmark = Benchmark()
    result_original = benchmark.evaluate(E, "'Before', {}".format(embed), print=False)
    result_hard = benchmark.evaluate(
        E_hard, "'Hard debiased', {}".format(embed), print=False
    )
    if E_soft:
        result_soft = benchmark.evaluate(
            E_soft, "'Soft debiased', {}".format(embed), print=False
        )
        results = [result_original, result_hard, result_soft]
        benchmark.pprint_compare(
            results, ["Before", "Hard-debiased", "Soft-debiased"], embed
        )
    # If E_soft is None, do not include soft debiasing in benchmarks
    else:
        results = [result_original, result_hard]
        benchmark.pprint_compare(results, ["Before", "Hard-debiased"], embed)


def main():
    # Print basic experiment information
    print_details()

    # For each embedding, do the experiments
    for embed in FLAGS.embeddings:
        print("\n" + "#" * 56)
        print("# " + f"Doing the {embed} embedding".center(53) + "#")
        print("#" * 56)

        # Load the embedding
        E = WordEmbedding(embed)
        # Load professions and gender related lists from
        # Bolukbasi et al. for word2vec
        gender_words, defs, equalize_pairs, profession_words = load_data(E.words)
        # Define gender direction with PCA
        v_gender = we.doPCA(defs, E).components_[0]

        # Bias without debiasing
        if not FLAGS.no_show:
            show_bias(E, v_gender, profession_words, info="with bias")

        # Hard debiasing
        E_hard = hard(E, gender_words, defs, equalize_pairs)
        if not FLAGS.no_show:
            show_bias(E_hard, v_gender, profession_words, info="hard debiased")

        E_soft = None
        # Only do soft debiasing for small embeddings
        if embed.split("_")[-1] != "large":
            # Soft debiasing
            E_soft = soft(E, embed, gender_words, defs)
            if not FLAGS.no_show:
                show_bias(E_soft, v_gender, profession_words, info="soft debiased")

        # Run the benchmarks if nescessary
        if not FLAGS.no_bench:
            run_benchmark(E, E_hard, E_soft, embed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        type=str,
        default=["word2vec_small", "glove_small", "fasttext_small"],
        nargs="+",
        choices=[
            "word2vec_small",
            "word2vec_large",
            "glove_small",
            "glove_large",
            "fasttext_small",
            "fasttext_large",
        ],
        help='Space separated list of embedding types. \
                    Embedding must be one of "word2vec_small", \
                    "word2vec_large", "glove_small", "glove_large", \
                    "fasttext_small", "fasttext_large".',
    )
    parser.add_argument(
        "--do_soft",
        action="store_true",
        help="If flag is set, does soft debiasing of each embedding \
                    from scratch. Otherwise load precomputed soft debiased \
                    embeddings.",
    )
    parser.add_argument(
        "--no_bench",
        action="store_true",
        help="If flag is set, does not perform the RG, WS, MSR and \
                    WEAT benchmarks.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="If flag is set, does not show analogies and \
                    occupational gender bias.",
    )

    FLAGS, _ = parser.parse_known_args()

    main()
