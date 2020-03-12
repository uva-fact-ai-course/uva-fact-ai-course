import argparse
import time
import os
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from data_loader import DataLoader
from model import LSTM
from bias_regularization import bias_regularization_term

from VGraph import VGraph
from modelsaver import (check_model_exists, save_current_state,
                        load_current_state, save_for_early_stopping)


def get_gendered_words(gender_file, corpus):
    """Creates a genderd and non-gendered word set

    gender files can be:
        wikitext-2 (or wiki)
        penn
        dm (or CNNDaily)
    """
    if gender_file == "wikitext-2":
        gender_file = "wiki"
    if gender_file == "CNNDaily":
        gender_file = "dm"
    filename = os.path.join("gender_pairs", gender_file + '-gender-pairs')

    female_words, male_words = [],[]
    with open(filename, 'r', encoding='utf-8') as f:
        gender_pairs = f.readlines()

    for gp in gender_pairs:
        f, m = gp.replace("\n", "").split()
        female_words.append(f)
        male_words.append(m)

    gender_words = set(female_words) | set(male_words)

    word2idx = corpus.dictionary.word2idx
    D = torch.LongTensor([[word2idx[wf], word2idx[wm]]
                          for wf, wm in zip(female_words, male_words)
                          if wf in word2idx and wm in word2idx])

    N = torch.LongTensor([idx for w, idx in word2idx.items()
                         if w not in gender_words])

    return D, N


def train(config, start_epoch=1, best_validation_loss=np.inf):
    """Trains AWD-LSTM model using parameters from config."""
    print(f'Training for {config.epochs} epochs using the {config.dataset}',
          f'dataset with lambda value of {config.encoding_lmbd}')

    device = torch.device(config.device)
    dataLoader = DataLoader(config.dataset,
                            config.batch_size,
                            device,
                            config.bptt)
    model = LSTM(embedding_size=config.embedding_size,
                 hidden_size=config.hidden_size,
                 lstm_num_layers=config.lstm_num_layers,
                 vocab_size=len(dataLoader.corpus.dictionary),
                 batch_size=config.batch_size,
                 dropoute=config.dropoute,
                 dropouti=config.dropouti,
                 dropouth=config.dropouth,
                 dropouto=config.dropouto,
                 weight_drop=config.weight_drop,
                 tie_weights=config.tie_weights,
                 device=device)

    # D is set of gendered words, N is neutral words (not entirely correct, but close enough)
    D, N = get_gendered_words(config.dataset, dataLoader.corpus)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)

    def using_asgd():
        """Checks if optimizer is using ASGD"""
        return 't0' in optimizer.param_groups[0]

    if not config.overwrite and check_model_exists(config):
        print("Loading model from precious state")
        model, optimizer, start_epoch, best_validation_loss = load_current_state(model, optimizer, config)
        if using_asgd():
            temp = torch.optim.ASGD(model.parameters(),
                                    lr=config.learning_rate,
                                    t0=0,
                                    lambd=0.,
                                    weight_decay=config.weight_decay)
            temp.load_state_dict(optimizer.state_dict())
            optimizer = temp
        print("start epoch", start_epoch)

    params = list(model.parameters()) + list(criterion.parameters())

    val_losses = deque(maxlen=config.nonmono)

    for e in range(start_epoch, config.epochs+1):
        epoch_done = False
        model.train()
        model.initialize_hidden()

        epoch_loss = 0 # Loss over the epoch
        n_batch = 0 # Number of batches that have been done
        t_start = time.time()
        print(f"starting epoch {e}/{config.epochs}")

        while not epoch_done:
            lr = optimizer.param_groups[0]['lr']

            # tr_batch, tr_labels are matrices with horizontal sequences.
            # seq_len is the sequence length in this iteration of the epoch,
            # see the openreviewpaper mentioned in the dataloader file
            tr_batch, tr_labels, seq_len, epoch_done = dataLoader.get_train_minibatch()

            # Rescale learning rate for sequence length
            optimizer.param_groups[0]['lr'] = lr * seq_len / config.bptt

            n_batch += 1
            model.detach_hidden() # Need to prevent improper backprop
            optimizer.zero_grad()

            out, _, lstm_raw_out, lstm_drop_out = model(tr_batch, return_out=True)
            loss = criterion(out.permute(0, 2, 1), tr_labels.t())

            # AR optimisation
            if config.alpha:
                loss += config.alpha * lstm_drop_out.pow(2).mean()

            # TAR optimisation
            if config.beta:
                loss += config.beta * (lstm_raw_out[1:] - lstm_raw_out[:-1]).pow(2).mean()

            # Encoding bias regularization
            if config.encoding_lmbd > 0:
                loss += bias_regularization_term(model.embed.weight, D, N, config.bias_variation, config.encoding_lmbd)

            # Decoding bias regularization
            if config.decoding_lmbd > 0:
                loss += bias_regularization_term(model.decoder.weight, D, N, config.bias_variation, config.decoding_lmbd)

            loss.backward()

            # Gradient clipping added to see effects. Turned off by default
            if config.clip: torch.nn.utils.clip_grad_norm_(params, config.clip)

            optimizer.step()

            # Add current loss to epoch loss
            epoch_loss += loss.item()

            # Return learning rate to default
            optimizer.param_groups[0]['lr'] = lr

            # Evaluate the training
            if n_batch % config.batch_interval == 0:
                cur_loss = epoch_loss / n_batch
                elapsed = float(time.time() - t_start)
                examples_per_second = n_batch / elapsed
                print('| epoch {:3d} | {:5d} batch | lr {:05.5f} | batch/s {:5.2f} | '
                      'train loss {:5.2f} | perplexity {:5.2f} |'.format(
                        e, n_batch, optimizer.param_groups[0]['lr'],
                        examples_per_second, cur_loss, np.exp(cur_loss)))

        print("Saving current model")
        save_current_state(model, optimizer, e, best_validation_loss, config)

        # Evaluate the model on the validation set for early stopping
        if e % config.eval_interval == 0:
            print('Evaluating on validation for early stopping criterion')
            test_done = False
            model.initialize_hidden()
            model.eval()
            epoch_loss = 0
            n_batch = 0
            tot_seq_len = 0
            while not test_done:
                n_batch += 1
                va_batch, va_labels, seq_len, test_done = dataLoader.get_validation_minibatch()
                tot_seq_len += seq_len
                out, _ = model(va_batch)
                model.detach_hidden()
                loss = criterion(out.permute(0, 2, 1), va_labels.t())
                epoch_loss += loss.item()
            cur_loss = epoch_loss / n_batch

            if best_validation_loss > cur_loss:
                print("best_validation_loss > cur_loss")
                best_validation_loss = cur_loss
                val_losses.append(cur_loss)
                save_for_early_stopping(model, config, best_validation_loss)

            print('| epoch {:3d} | lr {:05.5f} | validation loss {:5.2f} | perplexity {:5.2f} |'.format(
                        e, optimizer.param_groups[0]['lr'], cur_loss, np.exp(cur_loss)))

            if not config.no_asgd and not using_asgd() and (len(val_losses) == val_losses.maxlen and cur_loss > min(val_losses)):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=config.learning_rate, t0=0, lambd=0., weight_decay=config.weight_decay)

        # Evaluate the model on the test set
        if e % config.eval_interval == 0:
            print('Evaluating on test')
            test_done = False
            model.eval()
            model.initialize_hidden()
            epoch_loss = 0
            n_batch = 0
            while not test_done:
                n_batch += 1
                te_batch, te_labels, seq_len, test_done = dataLoader.get_test_minibatch()
                out, _ = model(te_batch)
                model.detach_hidden()
                loss = criterion(out.permute(0, 2, 1), te_labels.t())
                epoch_loss += loss.item()
            cur_loss = epoch_loss / n_batch

            print('| epoch {:3d} | lr {:05.5f} | test loss {:5.2f} | perplexity {:5.2f} |'.format(
                        e, optimizer.param_groups[0]['lr'], cur_loss, np.exp(cur_loss)))

    print(f'Training is done. Best validation loss: {best_validation_loss}, validation perplexity: {np.exp(best_validation_loss)}')


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name, 'penn', 'wikitext-2' or 'CNNDaily'")
    parser.add_argument('--bptt', type=int, default=70, help='Length of an input sequence')
    parser.add_argument('--hidden_size', type=int, default=1150, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=3, help='Number of LSTM layers in the model')
    parser.add_argument('--embedding_size', type=int, default=400, help='Number of LSTM layers in the model')
    parser.add_argument('--clip', type=float, default=0.25, help='Model gradient clipping value')
    parser.add_argument('--weight_drop', type=float, default=0.5, help='Weight drop for LSTM')
    parser.add_argument('--tie_weights', action='store_true', help='Tie embedding and decoder weights of LSTM')
    parser.add_argument('--alpha', type=float, default=2, help='LSTM output regularization')
    parser.add_argument('--beta', type=float, default=1, help='LSTM output slowness')
    parser.add_argument('--weight_decay', type=float, default=1.2e-6, help='Weight decay for optimizer')
    parser.add_argument('--no_asgd', action='store_true', help='Do not use ASGD')
    parser.add_argument('--nonmono', type=int, default=5, help='Non-monotonic parameter for ASGD')

    # Dropout params
    parser.add_argument('--dropoute', type=float, default=0.1, help='Dropout for embedding layer')
    parser.add_argument('--dropouti', type=float, default=0.65, help='Dropout for lstm input')
    parser.add_argument('--dropouth', type=float, default=0.3, help='Dropout for lstm hidden layer')
    parser.add_argument('--dropouto', type=float, default=0.4, help='Dropout for lstm output')

    # Debiasing params
    parser.add_argument('--encoding_lmbd', type=float, default=0, help='Encoding lambda')
    parser.add_argument('--decoding_lmbd', type=float, default=0, help='Decoding lambda')
    parser.add_argument('--bias_variation', type=float, default=0.5, help='Variation ratio for svd')

    # Training params
    parser.add_argument('--batch_size', type=int, default=40, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=30, help='Learning rate')

    parser.add_argument('--epochs', type=int, default=750, help='Number of epochs steps')
    parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs in between each test evaluation')
    parser.add_argument('--batch_interval', type=int, default=50, help='Number of batches in between each evaluation')

    parser.add_argument('--device', type=str, default='cuda:0', help="'cuda:0' or 'cpu'")

    parser.add_argument('--seed', type=int, default=1, help='Seed for reproducability')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite earlier saved model if it exists')
    parser.add_argument('--overview_file', type=str, default="model_overview", help='File which gives an overview of which model corresponds to which set of parameters')

    config = parser.parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed) # Safe to call even without cuda

    # Train the model
    train(config)
