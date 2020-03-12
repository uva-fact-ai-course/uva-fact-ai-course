import os
import numpy as np
import math
import json

from collections import Counter
from nltk import ngrams
from nltk.corpus import stopwords


class BiasCalculator:
    """Calculates bias scores according to the paper:

    Identifying and Reducing Gender Bias in Word-level Language Models (Bordia & Bowman)
    """
    def __init__(self, window=10, beta=.95, output_dir='./output', gender_pair_dir='./gender_pairs'):
        self.window = 2 * window + 1
        self.beta = beta

        self.stop_words = set(stopwords.words('english'))

        self.freq_counter = Counter()
        self.word_counter = Counter()
        self.fixed_cooccurrence_counter = Counter()
        self.infinite_cooccurrence_counter = Counter()
        self.vocab = set()
        self.fixed_cond_prob = dict()
        self.infinite_cond_prob = dict()
        self.fixed_bias_scores = dict()
        self.infinite_bias_scores = dict()

        self.output_dir = self.__check_directory(output_dir, create=True)
        self.gender_pair_dir = self.__check_directory(gender_pair_dir)

    def __check_directory(self, directory, create=False):
        """Check if directory exists."""
        abs_dir_path = os.path.abspath(directory)
        if not os.path.exists(abs_dir_path) or not os.path.isdir(abs_dir_path):
            print(f'Output directory {directory} does not exist')
            if create:
                print('Creating directory.')
                os.makedirs(abs_dir_path)
                print(f'New output directory {directory} created. Results will be saved to this directory.')
                return abs_dir_path
            else:
                raise ValueError(f'Dataset directory {directory} does not exist')

        return abs_dir_path

    @property
    def __skip_words(self):
        """Returns a list of words to skip in the corpus.

        Should return a set of stop words and gendered words
        """
        if hasattr(self, 'male_nouns'):
            return self.stop_words | set(self.male_nouns) | set(self.female_nouns)
        else:
            return self.stop_words

    def __split_text(self, text):
        """Split text based on End Of Sentence <EOS> tags."""
        if '<eos>' in text:
            sentences = [text.replace('<eos>', '')]
        else:
            sentences = text.split('\n')
        return [sentence.split() for sentence in sentences]

    def __set_gender_nouns(self):
        """Retrieve gender nouns from gender pair file."""
        pair_file = self.dataset_name + '-gender-pairs'
        pair_path = os.path.join(self.gender_pair_dir, pair_file)

        with open(pair_path, 'r', encoding='utf-8') as f:
            gender_pairs = [pair.split() for pair in f.readlines()]

        self.male_nouns, self.female_nouns = zip(*gender_pairs)

    def __set_frequency(self, sentences):
        """Find frequency of male and female gendered words."""
        m = 0
        f = 0

        for sentence in sentences:
            for word in sentence:
                if word in self.male_nouns:
                    m += 1

                if word in self.female_nouns:
                    f += 1

        self.freq_counter += Counter({'m': m, 'f':f})

    def __set_word_count(self, sentences):
        """Count frequency of words."""
        self.word_counter += Counter([word for sentence in sentences for word in sentence if word not in self.__skip_words])

    def __set_fixed_cooccurrence_count(self, sentences):
        """Find coocurrences in a fixed context."""
        for sentence in sentences:
            n_grams = ngrams(sentence, self.window, pad_left=True, pad_right=True)

            centre_pos = (self.window - 1) // 2

            for gram in n_grams:
                centre_word = gram[centre_pos]

                for word in gram:
                    if word in self.__skip_words:
                        continue

                    if word not in self.fixed_cooccurrence_counter:
                        self.fixed_cooccurrence_counter[word] = {'m': 0, 'f': 0}

                    if centre_word in self.male_nouns:
                        self.fixed_cooccurrence_counter[word]['m'] += 1

                    if centre_word in self.female_nouns:
                        self.fixed_cooccurrence_counter[word]['f'] += 1

    def __set_infinite_cooccurrence_count(self, sentences, beta):
        """Find coocurrences in an infinite context."""
        for sentence in sentences:
            for i, centre in enumerate(sentence):
                if centre in self.__skip_words:
                    continue

                if centre not in self.infinite_cooccurrence_counter:
                    self.infinite_cooccurrence_counter[centre] = {'m': 0, 'f': 0}

                for j, word in enumerate(sentence):
                    if i == j:
                        continue

                    distance = abs(i - j)

                    if word in self.male_nouns:
                        self.infinite_cooccurrence_counter[centre]['m'] += pow(beta, distance)

                    if word in self.female_nouns:
                        self.infinite_cooccurrence_counter[centre]['f'] += pow(beta, distance)

    def __set_vocab(self, sentences):
        """Set vocabulary to be used (all words minus skip words)."""
        self.vocab |= set([word for sentence in sentences for word in sentence])
        self.vocab -= self.__skip_words

    def __set_cond_prob(self, cooccurrence_counter, cond_prob):
        """Calculate conditional probabilities."""
        sum_male_coocurrence = sum([cooccurrence_counter[w]['m'] for w in cooccurrence_counter])
        sum_female_coocurrence = sum([cooccurrence_counter[w]['f'] for w in cooccurrence_counter])

        male_frequency = self.freq_counter['m']
        female_frequency = self.freq_counter['f']

        sum_frequency = sum(self.word_counter.values())

        for target_word in self.vocab:
            target_male_cooccurrence = cooccurrence_counter[target_word]['m']

            target_female_cooccurrence = cooccurrence_counter[target_word]['f']

            if not (target_male_cooccurrence and sum_male_coocurrence and male_frequency and sum_frequency):
                male_cond_prob = 0
            else:
                male_cond_prob = (target_male_cooccurrence / sum_male_coocurrence) / (male_frequency / sum_frequency)


            if not (target_female_cooccurrence and sum_female_coocurrence and female_frequency and sum_frequency):
                female_cond_prob = 0
            else:
                female_cond_prob = (target_female_cooccurrence / sum_female_coocurrence) / (female_frequency / sum_frequency)

            cond_prob[target_word] = {'m': male_cond_prob, 'f': female_cond_prob}

    def __calculate_mean_std(self, bias_scores):
        """Calculate the mean absolute and standard deviation of the bias scores."""
        abs_bias_scores = np.abs([val for val in bias_scores.values()])
        mean_bias = np.mean(abs_bias_scores)
        std_bias = np.std([val for val in bias_scores.values()])

        return mean_bias, std_bias

    def __configure_model_calculator(self, dataset_dir_path, model_dir=''):
        """Setup the model calculator (calculate conditional probabilities)."""
        self.__init__(self.window // 2, self.beta, self.output_dir, self.gender_pair_dir)

        self.__set_gender_nouns()

        model_dir_path = os.path.join(dataset_dir_path, model_dir)

        for root, _, files in os.walk(model_dir_path):
            root = os.path.abspath(root)
            for fname in files:
                basename, _ = os.path.splitext(fname)
                if basename.lower() == 'readme':
                    continue

                txt_path = os.path.join(root, fname)

                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    sentences = f.read()
                    sentences = self.__split_text(sentences)

                self.__set_vocab(sentences)
                self.__set_frequency(sentences)
                self.__set_word_count(sentences)
                self.__set_fixed_cooccurrence_count(sentences)
                self.__set_infinite_cooccurrence_count(sentences, self.beta)

        self.__set_cond_prob(self.fixed_cooccurrence_counter, self.fixed_cond_prob)
        self.__set_cond_prob(self.infinite_cooccurrence_counter, self.infinite_cond_prob)

        print(f'Calculator for {self.dataset_name}-{model_dir} configured.')

    def __calculate_model_bias(self, model_name):
        """Calculate the bias scores of the model (fixed and infinite context)."""
        fixed_bias_scores = dict()
        infinite_bias_scores = dict()

        gendered_nouns = set(self.female_nouns) | set(self.male_nouns)

        for word in self.vocab:
            if word in gendered_nouns or word in self.stop_words:
                continue

            if self.fixed_cond_prob[word]['f'] != 0 and self.fixed_cond_prob[word]['m'] != 0:
                fixed_bias_scores[word] = math.log(self.fixed_cond_prob[word]['f'] / self.fixed_cond_prob[word]['m'])

            if self.infinite_cond_prob[word]['f'] != 0 and self.infinite_cond_prob[word]['m'] != 0:
                infinite_bias_scores[word] =  math.log(self.infinite_cond_prob[word]['f'] / self.infinite_cond_prob[word]['m'])

        model_output_dir = os.path.join(self.output_dir, self.dataset_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        bias_scores_path = os.path.join(model_output_dir, f'{model_name}')

        with open(bias_scores_path, 'w', encoding='utf-8') as f:
            json.dump({
                'fixed': fixed_bias_scores,
                'infinite': infinite_bias_scores
            }, f)

        fixed_mean_bias, fixed_std_bias = self.__calculate_mean_std(fixed_bias_scores)
        infinite_mean_bias, infinite_std_bias = self.__calculate_mean_std(infinite_bias_scores)

        overview_output_path = os.path.join(self.output_dir, 'bias-scores')

        with open(overview_output_path, 'a', encoding='utf-8') as f:
            f.write(f'{self.dataset_name} {model_name} \n')
            json.dump({
                'fixed': {
                    'mean abs': fixed_mean_bias,
                    'std abs': fixed_std_bias
                },
                'infinite': {
                    'mean abs': infinite_mean_bias,
                    'std abs': infinite_std_bias
                }
            }, f)
            f.write('\n')

        print(f'Bias scores calculated for {self.dataset_name}-{model_name}.')

    def calculate_bias(self, training_input_dir, generated_input_dir):
        """Calculate the bias score of the given models in the generated folder."""
        training_input_dir_path = self.__check_directory(training_input_dir)
        generated_input_dir_path = self.__check_directory(generated_input_dir)

        root, dirs, _ = next(os.walk(training_input_dir_path))
        for dataset_dir in dirs:
            if dataset_dir == 'wikitext-2':
                self.dataset_name = 'wiki'
            else:
                self.dataset_name = dataset_dir

            dataset_dir_path = os.path.join(os.path.abspath(root), dataset_dir)
            print("dataset_dir_path", dataset_dir_path)
            self.__configure_model_calculator(dataset_dir_path)
            self.__calculate_model_bias('train')

        root, dirs, _ = next(os.walk(generated_input_dir_path))
        for dataset_dir in dirs:
            self.dataset_name = dataset_dir

            dataset_dir_path = os.path.join(os.path.abspath(root), dataset_dir)
            for model_dir in next(os.walk(dataset_dir_path))[1]:
                self.__configure_model_calculator(dataset_dir_path, model_dir)
                self.__calculate_model_bias(model_dir)


if __name__ == "__main__":
    calc = BiasCalculator()
    calc.calculate_bias('./data', './generated')
