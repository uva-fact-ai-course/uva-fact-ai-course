from preprocess import read_preprocessed_file, read_vocab
from collections import Counter
from scipy.stats import entropy
from numpy.linalg import norm
import os
import numpy as np
import spacy

en = spacy.load('en')


DEFAULT_MALE_NOUNS = {
    'gentleman', 'man', 'men', 'gentlemen', 'male', 'males', 'boy', 'boyfriend',
    'boyfriends', 'boys', 'he', 'his', 'him', 'husband', 'husbands'
}

DEFAULT_FEMALE_NOUNS = {
    'woman', 'women', 'ladies', 'female', 'females', 'girl', 'girlfriend',
    'girlfriends', 'girls', 'her', 'hers', 'lady', 'she', 'wife', 'wives'
}

DEFAULT_TARGET_POS = {'VERB', 'ADJ', 'ADV'}


def get_sentence_gender_cooccurrences(sent, target_pos=None, female_nouns=None, male_nouns=None):
    """
    Get gender cooccurrences for a sentence
    """
    # Handle defaults
    if not target_pos:
        target_pos = DEFAULT_TARGET_POS
    if not female_nouns:
        female_nouns = DEFAULT_FEMALE_NOUNS
    if not male_nouns:
        male_nouns = DEFAULT_MALE_NOUNS
    gendered_nouns = female_nouns | male_nouns

    female_cooccur = Counter()
    male_cooccur = Counter()

    # Check which genders are present in this sentence
    female = False
    male = False
    for w in sent:
        if w in female_nouns:
            female = True
        elif w in male_nouns:
            male = True

        if female and male:
            break

    # If no gendered words were found, return
    if not (female or male):
        return female_cooccur, male_cooccur


    # Find all of the cooccurences of each gender
    for w in en(' '.join(sent)):
        if (w.pos_ in target_pos) and (w.text not in gendered_nouns):
            item = (w.text, w.pos_)
            if female:
                female_cooccur[item] += 1

            if male:
                male_cooccur[item] += 1

    return female_cooccur, male_cooccur


def get_sentence_list_gender_cooccurrences(sentences, target_pos=None,
                                           female_nouns=None, male_nouns=None):
    """
    Get gender cooccurrences for a list of sentences
    """
    female_cooccur = Counter()
    male_cooccur = Counter()
    for sent in sentences:
        female_cooccur_sent, male_cooccur_sent \
            = get_sentence_gender_cooccurrences(sent, target_pos=target_pos,
                                                female_nouns=female_nouns,
                                                male_nouns=male_nouns)

        female_cooccur += female_cooccur_sent
        male_cooccur += male_cooccur_sent

    return female_cooccur, male_cooccur


def get_dataset_gender_cooccurrences(data_dir, vocab, target_pos=None,
                                     female_nouns=None, male_nouns=None):
    """
    Get gender cooccurrences for a preprocessed dataset
    """
    if type(vocab) == str:
        # Load vocab if a path was provided
        vocab = read_vocab(vocab)

    female_cooccur = Counter()
    male_cooccur = Counter()

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        # Read file
        sentences = read_preprocessed_file(fpath, vocab)
        female_cooccur_file, male_cooccur_file \
            = get_sentence_list_gender_cooccurrences(sentences, target_pos=target_pos)

        female_cooccur += female_cooccur_file
        male_cooccur += male_cooccur_file

    return female_cooccur, male_cooccur


def compute_gender_cooccurrance_bias(female_cooccur, male_cooccur):
    """
    Compute a bias metric similar to that in (Zhao et al. 2017)
    """
    bias_sum = 0.0
    bias_norm_sum = 0.0
    total_words = (set(female_cooccur.keys()) | set(male_cooccur.keys()))

    male_count = sum(male_cooccur.values())
    female_count = sum(female_cooccur.values())

    for k in total_words:
        p_w_given_male = male_cooccur[k] / male_count
        p_w_given_female = female_cooccur[k] / female_count

        # Ratio of joint probabilities
        bias_sum += male_cooccur[k] / (female_cooccur[k] + male_cooccur[k])

        # Ratio of conditional probabilities (conditioned on gender)
        bias_norm_sum += p_w_given_male / (p_w_given_male + p_w_given_female)

    bias = bias_sum / len(total_words)
    bias_norm = bias_norm_sum / len(total_words)

    return bias, bias_norm


def JSD(P, Q):
    """
    Compute Jensen-Shannon divergence
    """
    # Copied from https://stackoverflow.com/a/27432724/1260544
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def compute_gender_distribution_divergence(female_cooccur, male_cooccur):
    """
    Compute Jensen-Shannon divergence between the word probabilities conditioned
    on male and female
    """
    female_count = sum(female_cooccur.values())
    male_count = sum(male_cooccur.values())
    total_words = (set(female_cooccur.keys()) | set(male_cooccur.keys()))

    p_w_given_male = np.array([male_cooccur[k] / male_count for k in total_words])
    p_w_given_female = np.array([female_cooccur[k] / male_count for k in total_words])

    return JSD(p_w_given_female, p_w_given_male)

if __name__ == "__main__":
    # fc, mc = get_dataset_gender_cooccurrences('../awd-lstm/data/penn', [], )
