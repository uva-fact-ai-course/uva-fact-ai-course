import os
import numpy as np
import math
import json

from collections import Counter
from nltk import ngrams
from nltk.corpus import stopwords


class BetaCalculator:
    """Calculates beta score by linear regression."""
    def __init__(self, output_dir='./output'):
        self.output_dir = self.__check_directory(output_dir)

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

    def __load_base_scores(self, model_name):
        """Load bias scores of reference data set."""
        base_scores_file = os.path.join(self.dataset_dir_path, model_name)
        with open(base_scores_file, 'r', encoding='utf-8') as f:
            self.base_scores = json.load(f)

    def __calculate_model_beta(self, model_name):
        """Calculate beta scores of model."""
        new_scores_file = os.path.join(self.dataset_dir_path, model_name)
        with open(new_scores_file, 'r', encoding='utf-8') as f:
            self.new_scores = json.load(f)

        self.__calculate_beta(model_name, 'fixed')
        self.__calculate_beta(model_name, 'infinite')

    def __calculate_beta(self, model_name, bias_type):
        """Calculate beta score for model depending on type 'fixed' or 'infinite'"""
        intersection_keys = set(self.base_scores[bias_type].keys()) & set(self.new_scores[bias_type].keys())

        base_vec = np.array([self.base_scores[bias_type][k] for k in intersection_keys])
        new_vec = np.array([self.new_scores[bias_type][k] for k in intersection_keys])

        coef = np.polyfit(base_vec, new_vec, 1) # returns a list of coefficients, coef of highest degree first

        output_path = os.path.join(self.output_dir, 'beta-scores')

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f'{self.dataset_name} {model_name} {bias_type} \nbeta: {coef[0]}, c: {coef[1]} \n')

        print(f'Beta value for {self.dataset_name} {model_name} {bias_type} calculated.')

    def calculate_beta(self, input_dir):
        """Start calculation of beta scores."""
        input_dir_path = self.__check_directory(input_dir)

        root, dirs, _ = next(os.walk(input_dir_path))
        for dataset_dir in dirs:
            self.dataset_name = dataset_dir

            self.dataset_dir_path = os.path.join(os.path.abspath(root), dataset_dir)
            self.__load_base_scores('train')

            for model_name in next(os.walk(self.dataset_dir_path))[2]:
                if model_name == 'train':
                    continue
                self.__calculate_model_beta(model_name)

        print(f'Beta values for stored in {self.output_dir}.')


if __name__ == "__main__":
    calculator = BetaCalculator()
    calculator.calculate_beta('./output')