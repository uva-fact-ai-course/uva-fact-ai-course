
import os
import argparse
import string
import random

import torch


# def gen_model_identifier(stringlength=10):
#     """Generate model identifier"""
#     symbol_options = "".join(filter(str.isalnum, string.printable))
#     new_identifier =  ''.join(random.choice(symbol_options) for i in range(stringlength))
#     print("identifier for new model:", new_identifier)
#     return new_identifier



# def get_ordered_config_vals(config_dict):
#     """ returns values from the config as a list of strings, ordered

#     alphabetically by key of the config
#     """
#     keys = sorted(config_dict.keys())
#     sorted_attributes = [str(config_dict[key]) for key in keys]
#     return sorted_attributes

# def get_header_order(config_dict):
#     return sorted(config_dict.keys())


# def config_to_dict(config):
#     return dict(vars(config))


# class ModelSaver:
#     def __init__(self, config):
#         self.cfg_dict = config_to_dict(config)
#         self.overview_file = self.cfg_dict["overview_file"]
#         self.prepare_overview_files()
#         self.model_location = os.path.join("models", self.identifier)
#         self.early_stopping_epoch = 0
#         self.has_saved = False



#     def prepare_overview_files(self):

#         if not os.path.exists("models"):
#             os.mkdir("models")
#         if not os.path.exists("plot_info"):
#             os.mkdir("plot_info")

#         if not os.path.exists(self.overview_file):
#             with open(self.overview_file, "w+", encoding='utf-8') as f:
#                 f.write(",".join(["identifier", "early stopping epoch"] + get_header_order(self.cfg_dict)) + "\n")
#             self.identifier = gen_model_identifier()
#         else:
#             with open(self.overview_file, "r", encoding='utf-8') as f:
#                 modeldata = f.readlines()
#                 identifiers = {datarow.split(",")[0] for datarow in modeldata}
#                 new_identifier = gen_model_identifier()
#                 while new_identifier in identifiers:
#                     new_identifier = gen_model_identifier()
#                 self.identifier = new_identifier



#     def has_saved_once(self):
#         return self.has_saved

#     def add_info_to_overview(self, vgraphs):
#         """ Ads the epoch of last validation improvement to the overview file,
#         along with the rest of the info"""

#         for vgraph in vgraphs:
#             resdir = os.path.join("plot_info", self.identifier)
#             if not os.path.exists(resdir):
#                 os.mkdir(resdir)
#             fname = os.path.join(resdir, vgraph.title.replace(" ", "_"))
#             header = ",".join([vgraph.xlabel, vgraph.ylabel])
#             values = zip(vgraph.x, vgraph.y)

#             with open(fname, "w+", encoding='utf-8') as f:
#                 f.write(header + "\n")
#                 for elem in values:
#                     f.write(",".join(elem) + "\n")

#         with open(self.overview_file, "a", encoding='utf-8') as f:
#             f.write(",".join([self.identifier, str(self.early_stopping_epoch)] + \
#                                     get_ordered_config_vals(self.cfg_dict)) + "\n")

def get_model_name_config(config):
    """Teturns the name assigned to a model based on config"""
    if not os.path.exists("models"):
        os.mkdir("models")
    return os.path.join("models","_".join(["dataset", config.dataset,
                                           "encoding_lmbd", str(config.encoding_lmbd),
                                           "decoding_lmbd", str(config.decoding_lmbd),
                                           "ASGD", str(not config.no_asgd)]))


def check_model_exists(config):
    """Check if model exists."""
    return os.path.exists(get_model_name_config(config))


def save_current_state(model, optimizer, epoch, best_validation_loss, config):
    """ saves most recent temp network/weights on even and odd epochs,

    this is done to handle timeouts on LISA and/or other time-limited environments
    """
    with open(get_model_name_config(config) + "TEMP_epoch_counter", "w+", encoding='utf-8') as f:
        print("saving epoch", epoch)
        f.write(str(epoch))
    torch.save({"model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch,
                "best_validation_loss":best_validation_loss}, get_model_name_config(config)+ f"TEMP_{epoch % 2}")


def load_current_state(model, optimizer, config):
    """Load saved state into model."""
    with open(get_model_name_config(config) + "TEMP_epoch_counter", "r", encoding='utf-8') as f:
        current_epoch = int(f.readline())
    try:
        checkpoint = torch.load(get_model_name_config(config)+ f"TEMP_{current_epoch % 2}")
    except Exception:
        current_epoch -= 1
        checkpoint = torch.load(get_model_name_config(config)+ f"TEMP_{current_epoch % 2}")

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['best_validation_loss']

    model.train()
    return model, optimizer, current_epoch, loss


def save_for_early_stopping(model, config, best_validation_loss):
    """Save model for early stopping."""
    torch.save({"model":model.state_dict(),
                "best_validation_loss":best_validation_loss}, get_model_name_config(config))


def load_from_early_stopping(model, config):
    """ loads the final version of the model in evaluation mode."""
    checkpoint = torch.load(get_model_name_config(config))
    model.eval()
    return model.load_state_dict(checkpoint["model"])




# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--dataset', type=str, required=True, help="Dataset name, 'penn' or 'wikitext-2'")
#     parser.add_argument('--bptt', type=int, default=30, help='Length of an input sequence')
#     parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in the LSTM')
#     parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
#     parser.add_argument('--embedding_size', type=int, default=3, help='embedding size')
#     parser.add_argument('--overview_file', type=str, default="modelsaver_check", help='file which gives an overview of which model corresponds to which set of parameters')
#     config = parser.parse_args()

#     m = ModelSaver(config)
#     m.add_info_to_overview([])
