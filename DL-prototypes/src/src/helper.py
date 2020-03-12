"""
Code of function list_of_distances by Oscar Li
github.com/OscarcarLi/PrototypeDL
"""
import os
import argparse
import torch

def check_path(path):
    """
    input: path of a directory
    Creates the directory if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def list_of_distances(x_vector, y_vector):
    """
    Given a list of vectors, x_vector = [x_1, ..., x_n], and another list of vectors,
    y_vector = [y_1, ... , y_m], we return a list of vectors
            [[d(x_1, y_1), d(x_1, y_2), ... , d(x_1, y_m)],
             ...
             [d(x_n, y_1), d(x_n, y_2), ... , d(x_n, y_m)]],
    where the distance metric used is the sqared euclidean distance.
    The computation is achieved through a clever use of broadcasting.
    """
    x_reshape = torch.reshape(list_of_norms(x_vector), shape=(-1, 1))
    y_reshape = torch.reshape(list_of_norms(y_vector), shape=(1, -1))
    output = x_reshape + y_reshape - 2 * (x_vector @ y_vector.t())

    return output

def list_of_norms(x_vector):
    """
    x_vector is a list of vectors x = [x_1, ..., x_n], we return
        [d(x_1, x_1), d(x_2, x_2), ... , d(x_n, x_n)], where the distance
    function is the squared euclidean distance.
    """
    return (torch.pow(x_vector, 2)).sum(axis=1)

def str2bool(bool_str):
    """
    input: string representing a boolean value
    converts the string to boolean
    """
    if isinstance(bool_str, bool):
        return input
    if bool_str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')
