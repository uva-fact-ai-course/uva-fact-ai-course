###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
#import pdb
from log import init_console_logger
import logging

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='/scratch/sb6416/experiments/data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

female_words = ['actress','girl ','mother', 'she', 'her ', 'her', 'female', 'woman', 'women', 'daughter',
                    'daughters','spokeswoman','husband','queen','sister']

male_words = ['actor', 'boy', 'father', 'he', 'him ', 'his', 'male', 'man', 'men', 'son', 
              'sons', 'spokesman', 'wife', 'king', 'brother']
# Set the random seed manually for reproducibility.
for seed in range(500,510):
    base_rate= []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3")

    with open(args.checkpoint, 'rb') as f:
        model, _, _ = torch.load(f,map_location=lambda storage, loc: storage)
    
    model.eval()
    if args.model == 'QRNN':
        model.reset()

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = Variable(torch.rand(1, 1).mul(ntokens).long())#, volatile=True)  
    #print("Input: ",input)
    if args.cuda:
        input.data = input.data.cuda()
    with open(args.outf, 'w') as outf:
        
        we = 0
        fe_weight=0
        me_weight=0
        for i in range(args.words):
            out1, hidden, output = model(input, hidden)
            decoded = model.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
            final_output=decoded.view(output.size(0), output.size(1), decoded.size(1))
            word_weights = final_output.squeeze().data.div(args.temperature).exp().cpu()
            S = torch.sum(word_weights)
            word_weights = word_weights/S
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            for w in female_words:
                fe_weight+=word_weights[corpus.dictionary.word2idx[w]]
             
            for w in male_words:
                me_weight+=word_weights[corpus.dictionary.word2idx[w]]


            #outf.write(word + ('\n' if i % 20 == 19 else ' '))
            outf.write(word + ' ' +str(word_weights[word_idx].data.numpy()) +'\n')

    t        if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

        #print(we, fe_weight, me_weight)
        if(me_weight!=0):
            base_rate[i].append(fe_weight/me_weight)
            print(fe_weight/me_weight)
            
            