
import os
import numpy as np

from collections import Counter
import torch


class DataLoader:
    """Loads data from dataset and produces batches."""
    def __init__(self, dataset_name, batchsize, device, bptt, standard_deviation=5):
        """ Handles loading data

        : Arguments :

            bptt: backpropagation through time
                (# standard sequence length, see https://openreview.net/pdf?id=SyyGPP0TZ)

            batch_size: int,
                size of batch, is smaller at the end of epoch
            device: string, torch device,
                'cuda' or 'cpu'

        """
        self.standard_deviation = standard_deviation
        self.batch_size = batchsize
        self.bptt = bptt
        self.load_corpus(dataset_name)

        # Inside the paper we reproduce validation is not mentioned anywhere,
        # I keep validation on hand in case we decide we need it

        self.train_data = batchify(self.corpus.train, batchsize, device)
        self.validation_data = batchify(self.corpus.valid, batchsize, device)
        self.test_data = batchify(self.corpus.test, batchsize, device)

        self.max_train_index = self.train_data.size(0) - 1
        self.train_index = 0
        self.max_validation_index = self.validation_data.size(0) - 1
        self.validation_index = 0
        self.max_test_index = self.test_data.size(0) - 1
        self.test_index = 0
        self.epoch_done = False
        self.epoch_done_test = False
        self.epoch_done_validation = False

    def load_corpus(self, dataset_name):
        """Load corpus."""
        fn = "corpus.{}.data".format(dataset_name)
        if os.path.exists(fn):
            print('Loading cached dataset...')
            corpus = torch.load(fn)
        else:
            print('Producing dataset...')
            corpus = Corpus(dataset_name)
            torch.save(corpus, fn)
        self.corpus = corpus

    def reset_for_new_epoch(self, datasubset):
        """Reset loader for epoch."""
        if datasubset == "train":
            self.train_index = 0
            self.epoch_done = False
        elif datasubset == "test":
            self.test_index = 0
            self.epoch_done_test = False
        elif datasubset == "validation":
            self.validation_index = 0
            self.epoch_done_validation = False
        else:
            raise ValueError(f"got an invalid data subset to reset, got {datasubset}, expected 'test' or 'train'")

    def get_validation_minibatch(self):
        """See get_train_minibatch for return values & comments"""
        epoch_done = False
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, self.standard_deviation)))
        seq_len = min(seq_len, self.bptt + 5*self.standard_deviation)

        if self.max_validation_index - self.validation_index <= seq_len:
            seq_len = self.max_validation_index - self.validation_index
            self.epoch_done_validation = True
            epoch_done = True

        seq_len = min(seq_len, self.max_validation_index - self.validation_index)
        data = self.validation_data[self.validation_index:self.validation_index+seq_len]
        target = self.validation_data[self.validation_index+1:self.validation_index+1+seq_len]
        self.validation_index += seq_len

        if self.epoch_done_validation:
            self.reset_for_new_epoch("validation")

        return data, target, seq_len, epoch_done

    def get_test_minibatch(self):
        """See get_train_minibatch for return values & comments"""
        epoch_done = False
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, self.standard_deviation)))

        seq_len = min(seq_len, self.bptt + 5*self.standard_deviation)

        if self.max_test_index - self.test_index <= seq_len:
            seq_len = self.max_test_index - self.test_index
            self.epoch_done_test = True
            epoch_done = True

        seq_len = min(seq_len, self.max_test_index - self.test_index)
        data = self.test_data[self.test_index:self.test_index+seq_len]
        target = self.test_data[self.test_index+1:self.test_index+1+seq_len]
        self.test_index += seq_len

        if self.epoch_done_test:
            self.reset_for_new_epoch("test")

        return data, target, seq_len, epoch_done

    def get_train_minibatch(self):
        """Retrieve a minibatch for training purposes.

        returns:
            data: torch tensor with indices, usage:
                data[:,0] is a sequence, data[0,:] is garbage, make sure the
                network procecesses this correspondingly during training
            target: same format as data, just a single index further
                in the sequence
            seq_len: integer,
                sequence length of the current batch
            epoch_done: boolean, indicates whether all training data has been
                passed through
        """
        epoch_done = False
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        # (Was not mentioned in the openreview paper, nor mentioned in the paper
        # we're reproducing, but it was in the to be reproduced papers codebase)
        seq_len = max(5, int(np.random.normal(bptt, self.standard_deviation)))

        # in some very unlikely scenario, sequences can become too large for
        # memory to handle due to the randomness introduced above. We capped it
        # at 5 standard deviations above the original (when the dataloader)
        # value. This should not detract much from the purpose of variable
        # sequence length as the chance of the length falling outside this range
        # is negligable; see the Table of numerical values @
        # https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        seq_len = min(seq_len, self.bptt + 5*self.standard_deviation)

        # print(f'seq: {seq_len}, max: {self.max_train_index}, cur: {self.train_index}')

        if self.max_train_index - self.train_index <= seq_len:
            seq_len = self.max_train_index - self.train_index
            self.epoch_done = True
            epoch_done = True

        seq_len = min(seq_len, self.max_train_index - self.train_index)
        data = self.train_data[self.train_index:self.train_index+seq_len]
        target = self.train_data[self.train_index+1:self.train_index+1+seq_len]
        self.train_index += seq_len

        if self.epoch_done:
            self.reset_for_new_epoch("train")

        return data, target, seq_len, epoch_done

    def idx2words(self, indices):
        """
        Accepts a list of indices and returns a list of the correspondig words

        Essentially just a bypass function so we can work with only the
        dataloader as an interface for anything we'd want to do with the data
        """
        return [self.corpus.dictionary.idx2word[ind] for ind in indices]


class Dictionary(object):
    """Dictionary class to hold corpus word information."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """Holds corpus information."""
    def __init__(self, dataset):
        path = os.path.join('data', dataset)
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


def batchify(data, batch_size, device):
    """Prep data for batches and put it on the desired device

    trimming off remainders (mentioned below), and pre-packaging the data this
    way negates part of the bonus from the variable sequence length, but this is
    done in codebase of the paper we replicate so we kept this in to stay close
    to the source* in this respect

    *see language-model-bias-master/awd-lstm/utils

    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    data = data.to(device)
    return data


if __name__ == "__main__":
    bptt, batch_size = 70, 40
    d = DataLoader("penn", batch_size, "cpu", bptt)
    epoch_done = False
    i = 0
    processed = 0
    while not epoch_done:
        data, target, seq_len, epoch_done = d.get_train_minibatch()
        i += 1
        processed += seq_len
        print("iter",i, "processed",processed)
        print(data.size(), target.size())

    data, target, seq_len, all_test_data_retrieved = d.get_test_minibatch()
    print(data.size(), target.size())
    print(d.idx2words(data[:,1].tolist()))
    print(d.idx2words(target[:,0].tolist()))
    inp = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
    print(data.size())
    print(inp.size())
    print()
