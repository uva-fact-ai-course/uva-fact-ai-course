from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from BLEU import bleu_score

import seaborn as sns
import pandas

from models import EncoderRNN, AttnDecoderRNN

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--source', type=str, default="fra", help="Source language (fra = NMT, eng = autoencoder)")
parser.add_argument('--target', type=str, default="eng", help="Target language (only eng)")
parser.add_argument('--use_uniform', type=bool, default=False, help="Use uniform distr instead of default attention mechanism")
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help="model parameter, controls convergence speed vs stability")

config = parser.parse_args()

print('##########')
print('Running experiment on the ', end='')
print('autoencoder' if config.source == 'eng' else 'fra->eng NMT', end=' ')
print('model, using ', end='')
print('uniform' if config.use_uniform else 'learned', end=' ')
print('attention')
print('##########')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# start/end of sentence tokens
SOS_token = 0
EOS_token = 1

# class that represents a language (collection of sentences + vocabulary + index2word and viceversa)
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# read data
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

# to reduce the dataset size and simplify the attention mechanism
#, we select sentences that are at lost 10 words long and sentences
# where the english translation start with the following prefixes
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

# full dataset reading/filtering/etc
input_lang, output_lang, pairs = prepareData(config.target, config.source, True)
print('Random dataset pair:', end='\t')
print(random.choice(pairs))

# divide in test and train
test_split = int(len(pairs)*0.8)

pairs_train = pairs[:test_split]
pairs_test = pairs[test_split:]

# dictionary keeping a sentece + all its translations (necessary for BLEU) of test set
multi_pairs_test = {}
for source, target in pairs_test:
    if source in multi_pairs_test.keys():
        multi_pairs_test[source].append(target)
    else:
        multi_pairs_test[source] = [target]

# the following functions are needed to transform a pair of strings into a NN-readable tensor

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# tune between using the decoder output vs using the true outputs
# teacher forcing -> faster convergence but unstable
# default 0.5

teacher_forcing_ratio = config.teacher_forcing_ratio

# training function

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# keep track of time

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0  
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs_train))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

# run encoder/decoder on a sentence, outputs the translation sentence + attention weights

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs_test)
        print('SOURCE:\t\t', pair[0])
        print('TRUE TRANSLATION:\t', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('MODEL TRANSLATION:\t', output_sentence)
        print('')

# compute dataset BLEU

def datasetBleu(multi_pairs_test, encoder, decoder):
    cumulative_bleu = 0.
    for source, targets in multi_pairs_test.items():
        translation_list, _ = evaluate(encoder, decoder, source)
        # output is a list and last one is EOS: we convert to string
        translation = ' '.join(translation_list[:-1])
        cumulative_bleu += bleu_score(translation, targets)

    return cumulative_bleu / len(multi_pairs_test.keys())

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size, device).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, device=device, uniform_attention = config.use_uniform).to(device)

# train or load if possible

# create folder if not there!
import os
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

att_type = 'att' if not config.use_uniform else 'uni'
try:
    encoder1 = torch.load(f'./saved_models/{config.source}_encoder_{att_type}.pt')
    attn_decoder1 = torch.load(f'./saved_models/{config.source}_decoder_{att_type}.pt')
    print('models loaded.')
except:
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    print('done training.')
    torch.save(encoder1, f'./saved_models/{config.source}_encoder_{att_type}.pt')
    torch.save(attn_decoder1, f'./saved_models/{config.source}_decoder_{att_type}.pt')

encoder1.eval()
attn_decoder1.eval()

evaluated_bleu_score = datasetBleu(multi_pairs_test, encoder1, attn_decoder1)
print(f'BLEU score on the test set: {evaluated_bleu_score}')
file = open(f"{config.source}_bleu_{att_type}.txt", "w")
file.write(str(evaluated_bleu_score))
file.close()

print('Random qualitative evaluations on test set: ')
evaluateRandomly(encoder1, attn_decoder1)

# our experiment
# code here is taken and edited from the evaluate function

def attentionPermutationExperiment(encoder, decoder, pairs, max_length=MAX_LENGTH):
    per_word = list()
    avgs, maxs = list(), list()
    counter = 0
    for source, _ in pairs:
        input_tensor = tensorFromSentence(input_lang, source)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        median_error = torch.zeros(max_length)
        max_weight = torch.zeros(max_length)

        for di in range(max_length):
        	## FOR EACH DECODING STEP, RUN PERMUTATION EXPERIMENT! (see model code)
            decoder_output, decoder_hidden, decoder_attention, median_error[di], max_weight[di] = \
                decoder.permutation_experiment(decoder_input, decoder_hidden, encoder_outputs)

            per_word.append((max_weight[di].item(), median_error[di].item()))
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        
        print('Iteration ' + str(counter) + ' done.')
        counter = counter + 1
        maxs.append(median_error.max().item())
        avgs.append(median_error.mean().item())

    return maxs, avgs, per_word

############################## PLOT RESULTS 

# used to devide data into chunks to plot
def map_to(max_weight):
    if max_weight < 0.25:
        res = -1
    elif max_weight < 0.50:
        res = -2
    elif max_weight < 0.75:
        res = -3
    else:
        res = -4
    return res

if not config.use_uniform:
    try:
        chunks_data = np.load(f'experiment_results_{config.source}.npy')
        maxs_data = np.load(f'max_aggregated_experiment_results_{config.source}.npy')
        avgs_data = np.load(f'avg_aggregated_experiment_results_{config.source}.npy')
    except:
    	# results will contain tuples (sentence, max_weight, median_error)
        maxs, avgs, per_word = attentionPermutationExperiment(encoder1, attn_decoder1, pairs_test)
        # "bin" them with the function map_to 
        chunks_data = np.array([[map_to(max_weight), median_error] for max_weight, median_error in per_word])

        maxs_data, avgs_data = np.array(maxs), np.array(avgs)
        # save
        np.save(f'experiment_results_{config.source}.npy', chunks_data)
        np.save(f'max_aggregated_experiment_results_{config.source}.npy', maxs_data)
        np.save(f'avg_aggregated_experiment_results_{config.source}.npy', avgs_data)

    # turn into a dataframe for plotting reason
    data = pandas.DataFrame(maxs_data, columns=['max median error'])

    # plot!
    sns.violinplot(x=data['max median error'], cut=0.02, inner='quartile', color='cornflowerblue')#, data=data, scale='count')
    plt.title(f'max {config.source}->{config.target} permutation experiment')
    plt.subplots_adjust(top=0.88) # prevents title cut-off
    plt.savefig(f'max_aggr_{config.source}2{config.target}_permutation_exp.png')

    # turn into a dataframe for plotting reason
    data = pandas.DataFrame(avgs_data, columns=['avg median error'])

    # plot!
    plt.figure()
    sns.violinplot(x=data['avg median error'], cut=0.02, inner='quartile', color='cornflowerblue')#, data=data, scale='count')
    plt.title(f'avg {config.source}->{config.target} permutation experiment')
    plt.subplots_adjust(top=0.88) # prevents title cut-off
    plt.savefig(f'avg_aggr_{config.source}2{config.target}_permutation_exp.png')

    #### per word
    # turn into a dataframe for plotting reason
    data = pandas.DataFrame(chunks_data, columns=['max attention', 'median error'])
    d = {-1: '[0..0.25)', -2: '[0.25..0.5)', -3: '[0.5..0.75)', -4: '[0.75..1]'}
    data = data.replace(d)

    # plot!
    plt.figure()
    sns.violinplot(x='median error', y='max attention', data=data, cut=0.02, inner='quartile', color='cornflowerblue', scale='count')
    plt.title(f'{config.source}->{config.target} permutation experiment')
    plt.subplots_adjust(top=0.88, left=0.25) # prevents title cut-off
    plt.savefig(f'{config.source}2{config.target}_permutation_exp.png')