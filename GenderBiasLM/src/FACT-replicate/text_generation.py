"""
Text generation file.

This file generates new sentences sampled from the language model
"""

import argparse
import os
import torch
from torch.autograd import Variable

from modelsaver import load_from_early_stopping
from data_loader import Corpus
from model import LSTM

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
parser.add_argument('--dataset', type=str, required=True, help="Dataset name, 'penn', 'wikitext-2' or 'dm'")

parser.add_argument('--model_name', type=str, required=True, help='Name of the model')

parser.add_argument('--model_path', type=str, default='./models/', help='Path to find the model')

parser.add_argument('--generate_folder', type=str, default='./generated/', help='Folder that contains generated text folders')

parser.add_argument('--lmbda', type=float, required=True, help='Lambda value of the model')

parser.add_argument('--files', type=int, default=2000, help='Number of files to generate')

parser.add_argument('--words', type=int, default=500, help='Number of words per file to generate')

parser.add_argument('--seed', type=int, default=1111,  help='Random seed')

parser.add_argument('--device', type=str, default='cuda:0', help="'cuda:0' or 'cpu'")

parser.add_argument('--temperature', type=float, default=1.0, help='Temperature - higher will increase diversity')

parser.add_argument('--log_interval', type=int, default=100, help='Reporting interval')

config = parser.parse_args()

device = torch.device(config.device)


def load_corpus(dataset_name):
    """Loads corpus data."""
    fn = "corpus.{}.data".format(dataset_name)
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        print(dataset_name)
        corpus = Corpus(dataset_name)
        torch.save(corpus, fn)

    return corpus


datafolder = config.dataset
if datafolder == 'wikitext-2':
    datafolder = 'wiki'

folder = config.generate_folder
if not os.path.exists(folder):
    os.mkdir(folder)

folder = os.path.join(folder, datafolder)
if not os.path.exists(folder):
    os.mkdir(folder)

folder = os.path.join(folder, str(config.lmbda))
if not os.path.exists(folder):
    os.mkdir(folder)

# Creates config.files number of files, each starting with a different seed
# This seed is necessary to produce random input, but we still need reproducability
for j in range(config.files):
    torch.manual_seed(j + 500)
    torch.cuda.manual_seed(config.seed)
    output_file = os.path.join(folder, f'{config.model_name}_temp_{config.temperature}_{j}.txt')

    if config.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    corpus = load_corpus(config.dataset)
    ntokens = len(corpus.dictionary)

    model_path = os.path.join(config.model_path, config.model_name)

    model = LSTM(vocab_size=ntokens, device=device)

    print("model path:\n", model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.device))['model'])
    model.batch_size = 1

    model.eval()

    model.initialize_hidden()
    input = torch.rand(1, 1).mul(ntokens).long().to(device)

    with open(output_file, 'a', encoding='utf-8') as outf:
        outf.write(' ')
        for i in range(config.words):
            output, _ = model(input)
            word_weights = output.squeeze().data.div(config.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % config.log_interval == 0:
                print('| Generated {}/{} words'.format(i, config.words))
