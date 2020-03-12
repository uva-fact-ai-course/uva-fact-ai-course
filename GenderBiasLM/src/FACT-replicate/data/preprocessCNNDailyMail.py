
import tarfile
import os
import shutil
import random
from math import floor
import re
from unidecode import unidecode
import spacy

# python -m spacy download en_core_web_sm
en = spacy.load('en_core_web_sm')


def is_valid_token(w):
    """
    Returns True if a token is valid
    """
    return bool(re.search('[a-zA-Z0-9]+', w))


def transform_token(w):
    """
    Transforms a token by making lowercase, and for numeric tokens replaces
    digits with placeholders
    """
    return re.sub(r'[.\-\']+$', '',
                  re.sub(r'[.\-\']+', '',
                         re.sub(r'[^A-Za-z<>$.\-\']', '',
                                re.sub(r'\d+', '<NUM>',
                                       unidecode(w).lower()))))


def extract(filename):
    """Extract file from tar."""
    print("extracting", filename)
    with tarfile.open(filename) as ref:
        ref.extractall(filename.split(".")[0])


def subsample_files(directories, factor):
    """Subsample files by a factor based on sentences (stories)."""
    if not os.path.exists("subsampled"):
        os.mkdir("subsampled")
    for directory in directories:
        print("subsampling from", directory)
        for root, dir, files in os.walk(directory):
            filename_properties = [[root, dir, file] for file in files]
            print(root, dir, len(files))
        num_subsampled = int(len(filename_properties)/factor)
        filenames = [os.path.join(root,file) for root, dir, file in filename_properties]
        new_filenames = [os.path.join("subsampled", "".join(root.split(os.path.sep)) + "".join(file.split(os.path.sep))) for root, dir, file in filename_properties]
        subsampled_filenames = filenames[:num_subsampled]
        subsampled_new_filenames = new_filenames[:num_subsampled]
        for old, new in zip(subsampled_filenames, subsampled_new_filenames):
            shutil.copy(old, new)


def train_val_test_split(ratios, dir_name):
    """splits the stories into train.txt, val.txt and test.txt
    ratios (str): '<train ratio>:<validation ratio>:<test ratio>'
    example: ratios = '5:3:2'
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for f in ['train.txt', 'valid.txt', 'test.txt']:
        if not os.path.exists(os.path.join(dir_name, f)):
            temp = open(os.path.join(dir_name, f), 'w', encoding='utf-8')
            temp.close()

    ratios = ratios.split(':')
    total = sum(int(r) for r in ratios)
    train_ratio = int(ratios[0]) / total
    validation_ratio = int(ratios[1]) / total
    test_ratio = int(ratios[2]) / total

    for root, dir, files in os.walk('subsampled'):
        total_files = len(files)
        train_no = floor(train_ratio * total_files)
        val_no = floor(validation_ratio * total_files)
        test_no = total_files - train_no - val_no
        random.shuffle(files)
        for i, file in enumerate(files):
            with open(os.path.join('subsampled', file), 'r', encoding='utf-8') as f2:
                if i < train_no:
                    with open(os.path.join(dir_name, 'train.txt', encoding='utf-8'), 'a') as f:
                            file_text = '\n'.join([' '.join(line) for line in process(f2.read())])
                            f.write(file_text)
                            f.write('\n')

                elif i <  train_no + val_no:
                    with open(os.path.join(dir_name, 'valid.txt'), 'a' encoding='utf-8') as f:
                            file_text = '\n'.join([' '.join(line) for line in process(f2.read())])
                            f.write(file_text)
                            f.write('\n')
                else:
                    with open(os.path.join(dir_name, 'test.txt'), 'a', encoding='utf-8') as f:
                            file_text = '\n'.join([' '.join(line) for line in process(f2.read())])
                            f.write(file_text)
                            f.write('\n')


def process(text):
    """Preprocesses given text.

    Splits stories, only retains their contents,
    filters for alphanumerics, replaces numbers by a num tag,
    sets text to lowercase and non-alpha numericals become <unk>
    """
    # Remove any additional information e.g. "@highlights"
    main_text_body = text.split('\n@')[0]

    # Split up lines, and then break up lines into sentences
    sentences = []
    for line in main_text_body.split('\n\n'):
        sentences += list(en(line.strip('\n')).sents)

    sentence_tokens = []
    for sent in sentences:
        sent_tokens = []
        for w in sent:
            if not is_valid_token(w.text):
                continue
            w = transform_token(w.text)
            sent_tokens.append(w)
        if len(sent_tokens) > 1:
            sentence_tokens.append(sent_tokens)
    return sentence_tokens


def cnnDailymail_data_exists():
    """Checks if the cnn and dailymail datasets exist."""
    return os.path.exists("dailymail_stories.tgz") and os.path.exists("cnn_stories.tgz")


if __name__ == "__main__":
    if not cnnDailymail_data_exists():
        print("download both the CNN as well as the dailymail stories from:\nhttps://cs.nyu.edu/~kcho/DMQA/\n save them as the default filename in FACT-replicate/data")
        print("re-run cell when done")
    else:
        if not os.path.exists("dailymail_stories"):
            extract("dailymail_stories.tgz")

        if not os.path.exists("cnn_stories"):
            extract("cnn_stories.tgz")

        if not os.path.exists("subsampled"):
            subsample_files(["cnn_stories", "dailymail_stories"], 100)

        # added later, to create train, val and test:
        if not os.path.exists("dm"):
            train_val_test_split("12:1:1", "dm")
