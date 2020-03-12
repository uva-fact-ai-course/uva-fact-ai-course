import os

import numpy as np
import torch
import argparse
from model import LSTM
from data_loader import DataLoader


def get_perplexity(model, dataloader, batchsize, print_verbose=False):
    """ Calculate perplexity of a model using the following metric:

    Ppl = e^(total_loss / (sequence length * batch size))
    """
    print("starting to calculate perplexity")
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    test_done = False
    model.initialize_hidden()
    model.eval()
    epoch_loss = 0
    tot_seq_len = 0
    while not test_done:
        test_batch, test_labels, seq_len, test_done = dataloader.get_test_minibatch()
        tot_seq_len += seq_len
        out, _ = model(test_batch)
        model.detach_hidden()
        loss = criterion(out.permute(0, 2, 1), test_labels.t())
        epoch_loss += loss.item()

        if print_verbose:
            print("currently at sequence point", dataloader.test_index, "/", dataloader.max_test_index)
    sum_reduction_loss = epoch_loss / (tot_seq_len * batchsize)

    sum_red_perplexity = np.exp(sum_reduction_loss)

    return sum_red_perplexity


def setup_model(model_name, dataset_name, model_path, device):
    """Sets up language-model (LSTM) on device based on its designated filename."""
    device = torch.device(device)
    batch_size = 20
    data_loader = DataLoader(dataset_name, batch_size, device, 70)
    model = LSTM(vocab_size=len(data_loader.corpus.dictionary), device=device, \
                    batch_size=batch_size)

    model_path = os.path.join(model_path, model_name)
    print("loading model state_dict...")
    print("model_path", model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model'])
    return model, data_loader, batch_size


def calculate_perplexities(model_names, dataset_names_eval, model_path, device_str, print_verbose=False, overwrite=False):
    """Calculate the perplexity."""
    model_path = os.path.join(os.path.curdir, "models")
    perplexities = []

    header = "dataset, lambda-value, perplexity"

    if not os.path.exists("perplexity_result_file.txt"):
        print("creating new file: perplexity_result_file.txt")
        print("file does not exist yet")
        with open("perplexity_result_file.txt", "w") as f:
            f.write(header + "\n")

    if overwrite:
        print("overwriting")
        with open("perplexity_result_file.txt", "w") as f:
            f.write(header + "\n")

    for i, model_name in enumerate(model_names):
        model_name_info = model_name.split("_")
        dataset = model_name_info[1]
        if dataset == "wikitext-2":
            dataset = "wiki"
        lambda_ = model_name_info[4]
        model_name = model_names[i]
        dataset = dataset_names_eval[i]
        print(dataset)

        model, data_loader, batch_size = setup_model(model_name, dataset_names_eval[i], model_path, device_str)

        # if you want elaborate terminal feedback (which you don't), set print_verbose to True
        print(model_name)
        perplexities.append(get_perplexity(model, data_loader, batch_size, print_verbose=print_verbose))

        print("new found perplexity:", perplexities[-1], "for model", model_name)

        with open("perplexity_result_file.txt", "a") as f:
            model_name_info = model_name.split("_")
            dataset = model_name_info[1]
            lambda_ = model_name_info[4]
            f.write(model_name + ", ".join([dataset, lambda_, str(perplexities[-1])]) + "\n")


if __name__ == "__main__":
    print("starting main for some reason")
