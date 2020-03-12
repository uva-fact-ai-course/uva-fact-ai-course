import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from log import init_console_logger
import logging
import data
import model

from utils import batchify, get_batch, repackage_hidden

#parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
#parser.add_argument('--data', type=str, default='data/penn/',
 #                   help='location of the data corpus')
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')

randomhash = ''.join(str(time.time()).split('.'))


parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--bias_reg_encoder', action='store_true',
                    help='use bias regularization encoder')
parser.add_argument('--bias_reg_decoder', action='store_true',
                    help='use bias regularization decoder')
parser.add_argument('--bias_reg_de_factor', type=float, default=1.0,
                    help='bias regularization decoder loss weight factor')
parser.add_argument('--bias_reg_en_factor', type=float, default=1.0,
                    help='bias regularization encoder loss weight factor')
parser.add_argument('--bias_reg_var_ratio', type=float, default=0.5,
                    help=('ratio of variance used for determining size of gender'
                          'subspace for bias regularization'))
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--data', type=str, default='/beegfs/yw3004/projects/language_bias/data/penn/',
                    help='location of the data corpus')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--gender_pair_file', type=str, default=None,
                    help=('debias using these gender pairs'))
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--patience', type=int, default=-1,
                    help='Early stopping patience. If -1, early stopping is not used')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--unnorm-bias', dest='norm_bias', action='store_false',
                    help='If set, do not normalize embedding weights before computing bias score')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')



args = parser.parse_args()
args.tied = True



LOGGER = logging.getLogger('training')
LOGGER.setLevel(logging.DEBUG)
init_console_logger(LOGGER, verbose=True)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        LOGGER.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    LOGGER.info('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    LOGGER.info('Producing dataset...')
    print(args.data)
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)




filename  = os.path.join("gender_pairs", args.gender_pair_file + '-gender-pairs')

female_words, male_words =[],[]
with open(filename,'r') as f:
    gender_pairs = f.readlines()

for gp in gender_pairs:
    f,m=gp.split()
    female_words.append(f)
    male_words.append(m)

gender_words = set(female_words) | set(male_words)

#print(female_words)
word2idx = corpus.dictionary.word2idx
D = torch.LongTensor([[word2idx[wf], word2idx[wm]]
                       for wf, wm in zip(female_words, male_words)
                       if wf in word2idx and wm in word2idx])

# Probably will want to make this better
N = torch.LongTensor([idx for w, idx in word2idx.items() if w not in gender_words])

if args.cuda:

    D = D.cuda()
    N = N.cuda()

eos_idx = corpus.dictionary.word2idx['<eos>']


def bias_regularization_encoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term
    """
    W = model.encoder.weight
    if norm:
        W = W / model.encoder.weight.norm(2, dim=1).view(-1, 1)

    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size()[0]):
        idxs = D[idx].view(-1)
        u = W[idxs[0],:]
        v = W[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)

    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S**2

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)

    # Get first k components to for gender subspace
    B = V[:, :k_idx.data[0]+1]
    loss = torch.matmul(W[N], B).norm(2) ** 2

    return lmbda * loss

def bias_regularization_decoder(model, D, N, var_ratio, lmbda, norm=True):
    """
    Compute bias regularization loss term
    """
    W = model.decoder.weight
    if norm:
        W = W / model.decoder.weight.norm(2, dim=1).view(-1, 1)

    C = []
    # Stack all of the differences between the gender pairs
    for idx in range(D.size()[0]):
        idxs = D[idx].view(-1)
        u = W[idxs[0],:]
        v = W[idxs[1],:]
        C.append(((u - v)/2).view(1, -1))
    C = torch.cat(C, dim=0)

    # Get prinipal components
    U, S, V = torch.svd(C)

    # Find k such that we capture 100*var_ratio% of the gender variance
    var = S**2

    norm_var = var/var.sum()
    cumul_norm_var = torch.cumsum(norm_var, dim=0)
    _, k_idx = cumul_norm_var[cumul_norm_var >= var_ratio].min(dim=0)

    # Get first k components to for gender subspace
    B = V[:, :k_idx.data[0]+1]
    loss = torch.matmul(W[N], B).norm(2) ** 2

    return lmbda * loss





eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    LOGGER.info('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, model.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    LOGGER.info('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
LOGGER.info('Args:', args)
LOGGER.info('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden, _, _ = model(data, hidden, return_h=True)
        total_loss +=  len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    #hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

        if args.bias_reg_encoder:
            bias_loss = bias_regularization_encoder(model, D, N, args.bias_reg_var_ratio, args.bias_reg_en_factor, norm=args.norm_bias)
            loss = loss + bias_loss ;

        if args.bias_reg_decoder:
            bias_loss = bias_regularization_decoder(model, D, N, args.bias_reg_var_ratio, args.bias_reg_de_factor, norm=args.norm_bias)
            loss = loss + bias_loss ;


        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            LOGGER.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
epochs_since_best_val_set = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            LOGGER.info('-' * 89)
            LOGGER.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            LOGGER.info('-' * 89)

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            if val_loss2 < stored_loss:
                model_save(args.save)
                LOGGER.info('Saving Averaged!')
                stored_loss = val_loss2
                epochs_since_best_val_set=0
            else:
            # Early stopping
                epochs_since_best_val_set += 1
                if args.patience > 0 and epochs_since_best_val_set >= args.patience:
                    LOGGER.info("Early stopping reached")
                    break


        else:
            val_loss = evaluate(val_data, eval_batch_size)
            LOGGER.info('-' * 89)
            LOGGER.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            LOGGER.info('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                LOGGER.info('Saving model (new best validation)')
                stored_loss = val_loss
                epochs_since_best_val_set = 0
            else:
            # Early stopping
                epochs_since_best_val_set += 1
                if args.patience > 0 and epochs_since_best_val_set >= args.patience:
                    LOGGER.info("Early stopping reached")
                    break


            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                LOGGER.info('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                LOGGER.info('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                LOGGER.info('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)


except KeyboardInterrupt:
    LOGGER.info('-' * 89)
    LOGGER.info('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
LOGGER.info('=' * 89)
LOGGER.info('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
LOGGER.info('=' * 89)
