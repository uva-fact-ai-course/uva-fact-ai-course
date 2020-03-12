""" This file contains code to run the experiments shown in
the reproduction of the paper "explanations based on the missing".
"""
import matplotlib.pyplot as plt
import numpy as np

import os
import torch
import argparse

from cem.datasets.mnist import MNIST
from cem.datasets.fashion_mnist import FashionMNIST

from cem.models.cae_model import CAE
from cem.models.conv_model import CNN

from cem.train import train_ae, train_cnn
from cem.cem import ContrastiveExplanationMethod

# set random seeds for reproducability
# (although the CEM is fully determininstic)
torch.manual_seed(0)
np.random.seed(0)


def main(args):

    if args.verbose:
        print("initialising classifier...")
    classifier = CNN(device=args.device)
    if args.verbose:
        print("loading classifier weights...")
    train_cnn(
        classifier,
        dataset=None,
        load_path=args.cnn_load_path,
        device=args.device)

    if not args.no_cae:
        if args.verbose:
            print("initialising autoencoder...")
        autoencoder = CAE(device=args.device)
        if args.verbose:
            print("loading autoencoder weights...")
        train_ae(
            autoencoder,
            dataset=None,
            load_path=args.cae_load_path,
            device=args.device)

    if args.verbose:
        print("loading dataset: {}".format(args.dataset))
    if args.dataset == "MNIST":
        try:
            dataset = MNIST()
        except RuntimeError:
            dataset = MNIST(download=True)

    elif args.dataset == "FashionMNIST":
        try:
            dataset = FashionMNIST()
        except RuntimeError:
            dataset = FashionMNIST(download=True)
    else:
        raise ValueError(
            "Incorrect dataset specified, please choose either MNIST or " +
            "FashionMNIST."
        )

    if args.verbose:
        print("initialising ContrastiveExplanationMethod...")

    CEM = ContrastiveExplanationMethod(
        classifier,
        autoencoder,
        iterations=args.iterations,
        n_searches=args.n_searches,
        kappa=args.kappa,
        gamma=args.gamma,
        beta=args.beta,
        learning_rate=args.learning_rate,
        c_init=args.c_init,
        c_converge=args.c_converge,
        verbose=args.verbose,
        print_every=args.print_every,
        input_shape=args.input_shape,
        device=args.device
    )

    if args.verbose:
        print("obtaining sample...")

    sample = dataset.get_sample_by_class(class_label=args.sample_from_class)

    if args.verbose:
        print("starting search...")
    delta = CEM.explain(sample, mode=args.mode)

    if delta is None:
        print("no solution found...")
        return

    before = torch.argmax(classifier(sample.view(-1, *args.input_shape)))

    if args.verbose:
        print("solution found!")

    print("original image classified as: {}".format(before))

    if args.mode == "PP":
        after = np.argmax(
            classifier(sample - delta.view(1, 28, 28)).detach().cpu()
        ).item()
        print("pertinent positive classified as: {}".format(after))
    elif args.mode == "PN":
        after = np.argmax(
            classifier(delta.view(-1, 1, 28, 28)).detach().cpu()
        ).item()
        print("image with pertinent negative added classified as: {}".format(
            after))

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        sharex=True,
        sharey=True,
        figsize=(10, 5)
    )

    ax1.imshow(sample.squeeze(), cmap="gray")
    ax1.title.set_text("original image")
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.imshow(delta.view(28, 28), cmap="gray")
    ax2.title.set_text("image with perturbation")
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    if args.mode == "PP":
        ax3.imshow(sample.squeeze() - delta.view(28, 28), cmap="gray")
        ax3.title.set_text("pertinent positive")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
    elif args.mode == "PN":
        ax3.imshow(delta.view(28, 28) - sample.squeeze(), cmap="gray")
        ax3.title.set_text("pertinent negative")
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

    plt.show()

    if not args.discard_images:
        # save the created images
        dirname = "saved_perturbations/{}-mode-{}-kappa-{}-gamma-{}".format(
            args.dataset, args.mode, args.kappa, args.gamma)
        os.makedirs(dirname, exist_ok=True)

        plt.imsave(
            dirname +
            "/orig-class-{}-before-{}-after-{}.png".format(
                args.sample_from_class, before, after),
            sample.squeeze(), cmap="gray")
        if args.mode == "PP":
            plt.imsave(
                dirname +
                "/pp-class-{}-before-{}-after-{}.png".format(
                    args.sample_from_class, before, after),
                sample.squeeze() - delta.view(28, 28), cmap="gray")
        elif args.mode == "PN":
            plt.imsave(
                dirname +
                "/pert-class-{}-before-{}-after-{}.png".format(
                    args.sample_from_class, before, after),
                delta.view(28, 28), cmap="gray")
            plt.imsave(
                dirname +
                "/pn-class-{}-before-{}-after-{}.png".format(
                    args.sample_from_class, before, after),
                delta.view(28, 28) - sample.squeeze(), cmap="gray")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Specify model and dataset to evaluate on
    parser.add_argument(
        "-dataset",
        help="choose a dataset (MNIST or FashionMNIST) to apply the " +
        "contrastive explanation method to.",
        type=str, default="MNIST")
    parser.add_argument(
        "-cnn_load_path",
        help="path to load classifier weights from.",
        default="./cem/models/saved_models/mnist-cnn.h5")
    parser.add_argument(
        "--no_cae",
        help="disable the autoencoder",
        action="store_true", default=False)
    parser.add_argument(
        "-cae_load_path",
        help="path to load autoencoder weights from.",
        default="./cem/models/saved_models/mnist-cae.h5")
    parser.add_argument(
        "-sample_from_class",
        help="specify which class to sample from for pertinent negative" +
        " or positive",
        default=3, type=int)
    parser.add_argument(
        "--discard_images",
        help="specify whether or not to save the created images",
        action="store_true", default=False)

    # Specify CEM optional arguments
    parser.add_argument(
        "-mode",
        help="Either PP for pertinent positive or PN for pertinent negative.",
        type=str, default="PN")
    parser.add_argument(
        "-kappa",
        help="kappa value used in the CEM attack loss.",
        type=float, default=10.0)
    parser.add_argument(
        "-beta",
        help="beta value used as L1 regularisation coefficient.",
        type=float, default=0.1)
    parser.add_argument(
        "-gamma",
        help="gamma value used as reconstruction regularisation coefficient",
        type=float, default=1.0)
    parser.add_argument(
        "-c_init",
        help="initial c value used as regularisation coefficient for the " +
        "attack loss",
        type=float, default=10.0)
    parser.add_argument(
        "-c_converge",
        help="c value to amend the value of c towards if no solution has " +
        "been found in the current iterations",
        type=float, default=0.1)
    parser.add_argument(
        "-iterations",
        help="number of iterations per search",
        type=int, default=1000)
    parser.add_argument(
        "-n_searches",
        help="number of searches",
        type=int, default=9)
    parser.add_argument(
        "-learning_rate",
        help="initial learning rate used to optimise the slack variable",
        type=float, default=0.01)
    parser.add_argument(
        "-input_shape",
        help="shape of a single sample, used to reshape input for " +
        "classifier and autoencoder input",
        type=tuple, default=(1, 28, 28))

    parser.add_argument(
        "--verbose",
        help="print loss information during training",
        action='store_true', default=False)
    parser.add_argument(
        "-print_every",
        help="if verbose mode is enabled, interval to print the current loss",
        type=int, default=100)
    parser.add_argument(
        "-device",
        help="device to run experiment on",
        type=str, default="cpu")

    args = parser.parse_args()
    main(args)
