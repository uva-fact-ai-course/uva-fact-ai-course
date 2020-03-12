import argparse
import logging
import os
import sys

import coloredlogs

import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import datasets.utils
import utils
from model import Predictor, ImagePredictor, Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion, metric, adversary, optimizer_A,
          scheduler, alpha, device, config):
    """
    Performs training of the models

    Args:
        dataloader_train (DataLoader): DataLoader for training
        dataloader_val (DataLoader): DataLoader for validation
        predictor (nn.Module): Predictor model
        optimizer_P (Optimizer): Optimizer for the predictor
        criterion (func): Loss criterion
        metric (func): Metric for evaluation
        adversary (nn.Module): Adversary model
        optimizer_A (Optimizer): Optimizer for the adversary
        scheduler (func): Scheduler for the learning rate
        alpha (float): Alpha hyperparameter for adversarial training
        device (torch.device): Device to train on
        config (dict): Dictionary containing number of epochs ('n_epochs'), whether to use the validation set ('val'),
        whether to debias ('debias'), the dataset name ('dataset_name')

    Returns: None

    """
    av_train_losses_P, av_train_losses_A, av_val_losses_P, av_val_losses_A = [], [], [], []
    train_scores_P, train_scores_A, val_scores_P, val_scores_A = [], [], [], []

    for epoch in range(config.n_epochs):

        # Forward (and backward when train=True) pass of the full train set
        train_losses_P, train_losses_A, labels_train_dict, protected_train_dict = utils.forward_full(dataloader_train,
                                                                                                     predictor,
                                                                                                     adversary,
                                                                                                     criterion,
                                                                                                     device,
                                                                                                     config.dataset_name,
                                                                                                     optimizer_P,
                                                                                                     optimizer_A,
                                                                                                     scheduler,
                                                                                                     train=True,
                                                                                                     alpha=alpha)


        # Store average training losses of predictor after every epoch
        av_train_losses_P.append(np.mean(train_losses_P))

        # Store train metrics of predictor after every epoch
        train_score_P = metric(labels_train_dict['true'], labels_train_dict['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor score [train] = {:.3f}'.format(epoch + 1, config.n_epochs, np.mean(train_losses_P),
                                                              train_score_P))
        train_scores_P.append(train_score_P)

        # Store train metrics of adversary after every epoch, if applicable
        if config.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            train_score_A = metric(protected_train_dict['true'], protected_train_dict['pred'])
            logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
                        'adversary score [train] = {:.3f}'.format(epoch + 1, config.n_epochs, np.mean(train_losses_A),
                                                                  train_score_A))
            train_scores_A.append(train_score_A)

        # Evaluate on validation set after every epoch, if applicable
        if config.val and dataloader_val is not None:
            with torch.no_grad():
                # Forward pass of full validation set
                val_losses_P, val_losses_A, labels_val_dict, protected_val_dict, _ = utils.forward_full(dataloader_val,
                                                                                                        predictor,
                                                                                                        adversary,
                                                                                                        criterion,
                                                                                                        device,
                                                                                                        config.dataset_name,
                                                                                                        optimizer_P,
                                                                                                        optimizer_A,
                                                                                                        scheduler)

            # Store average validation losses of predictor after every epoch
            av_val_losses_P.append(np.mean(val_losses_P))

            # Store validation metrics of predictor
            val_score_P = metric(labels_val_dict['true'], labels_val_dict['pred'])
            logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                        'predictor score [val] = {:.3f}'.format(epoch + 1, config.n_epochs, np.mean(val_losses_P),
                                                                val_score_P))

            # Store validation metrics of adversary, if applicable
            if config.debias:
                val_score_A = metric(protected_val_dict['true'], protected_val_dict['pred'])
                logger.info('Epoch {}/{}: adversary loss [val] = {:.3f}, '
                            'adversary score [val] = {:.3f}'.format(epoch + 1, config.n_epochs, np.mean(val_losses_A),
                                                                    val_score_A))

    logger.info('Finished training')

    # Plot scores and loss curves
    logger.info('Generating plots')
    utils.plot_learning_curves((av_train_losses_P, train_scores_P, av_val_losses_P, val_scores_P),
                               (av_train_losses_A, train_scores_A, av_val_losses_A, val_scores_A),
                               config.dataset_name)
    if config.dataset_name == 'crime' and config.val:
        utils.make_coplot(protected_val_dict, labels_val_dict)

    # Save the model
    os.makedirs(config.save_model_to, exist_ok=True)
    torch.save(predictor.state_dict(),
               config.save_model_to + "pred_debiased_" + str(config.debias) + "_" + str(config.dataset_name) + "_seed_" + str(
                   config.seed))
    if config.debias:
        torch.save(adversary.state_dict(), config.save_model_to + "adv_seed_" + str(config.seed))


def test(dataloader_test, predictor, adversary, criterion, metric, device, dataset_name, show_logs=True):
    """
    Performs testing with the models

    Args:
        dataloader_test (DataLoader): Dataloader for the test set
        predictor (nn.Module): Predictor model
        adversary (nn.Module): Adversary model
        criterion (func): Loss criterion
        metric (func): Metric for evaluation
        device (torch.device): Device to train on
        dataset_name (str): Name of the dataset
        show_logs (bool): Whether to print logs or not (default value True)

    Returns: Results on the test set

    """
    # Disable logging if show_logs is False
    if not show_logs:
        logging.disable(sys.maxsize)

    # Forward pass on the test set
    with torch.no_grad():
        test_losses_P, test_losses_A, labels_test_dict, protected_test_dict, pred_y_prob = utils.forward_full(dataloader_test,
                                                                                                              predictor, adversary, criterion, device, dataset_name)
        if dataset_name != 'crime':
            mutual_info = utils.mutual_information(protected_test_dict["true"], labels_test_dict['pred'], labels_test_dict['true'])

    # Compute the test metric with the predictor output
    test_score_P = metric(labels_test_dict['true'], labels_test_dict['pred'])
    logger.info('Predictor score [test] = {}'.format(test_score_P))

    # Compute the test metric with the adversary output
    if adversary is not None:
        test_score_A = metric(protected_test_dict['true'], protected_test_dict['pred'])
        logger.info('Adversary score [test] = {}'.format(test_score_A))

    # Compute additional metrics depending on the dataset
    if dataset_name == 'adult':
        neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr = utils.calculate_metrics(
            labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'], dataset_name)
        logger.info('Confusion matrix for the negative protected label: \n{}'.format(neg_confusion_mat))
        logger.info('FPR: {}, FNR: {}'.format(neg_fpr, neg_fnr))
        logger.info('Confusion matrix for the positive protected label: \n{}'.format(pos_confusion_mat))
        logger.info('FPR: {}, FNR: {}'.format(pos_fpr, pos_fnr))
        return test_score_P, neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr, mutual_info

    elif dataset_name == 'images':
        neg_prec, neg_recall, neg_fscore, neg_support, neg_auc, pos_prec, pos_recall, pos_fscore, pos_support, pos_auc, avg_dif, avg_abs_dif = utils.calculate_metrics(
            labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'], dataset_name, pred_probs=pred_y_prob)
        logger.info(f'Negative protected variable (men): precision {neg_prec}, recall {neg_recall}, F1 {neg_fscore}, support {neg_support}, AUC {neg_auc}.')
        logger.info(f'Positive protected variable (women): precision {pos_prec}, recall {pos_recall}, F1 {pos_fscore}, support {pos_support}, AUC {pos_auc}.')
        logger.info(f'Average difference between conditional probabilities: {avg_dif}')
        logger.info(f'Average absolute difference between conditional probabilities: {avg_abs_dif}')
        return test_score_P, neg_auc, pos_auc

    elif dataset_name == 'crime':
        logger.info('Generating conditional plot')
        utils.make_coplot(protected_test_dict, labels_test_dict)
        return test_score_P


if __name__ == "__main__":

    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--predictor_lr', type=float, default=0.001,
                        help='Predictor learning rate')
    parser.add_argument('--adversary_lr', type=float, default=0.001,
                        help='Adversary learning rate')
    parser.add_argument('--debias', action='store_true',
                        help='Whether to perform debiasing or not')
    parser.add_argument('--val', action="store_true",
                        help='Whether to use a validation set during training')
    parser.add_argument('--dataset_name', type=str, default='adult',
                        help='Name of the dataset to be used: adult, crime, images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed to train with')
    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')
    parser.add_argument('--lr_scheduler', choices=['exp', 'lambda'], default='exp',
                        help='Learning rate scheduler to use')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Value of alpha for the adversarial training')

    # Parse arguments
    args = parser.parse_args()

    # Display the configuration
    logger.info('Using configuration {}'.format(vars(args)))

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # Set seed if given
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data and obtain dataloaders for train, validation and test
    logger.info('Loading the dataset')
    dataloader_train, dataloader_val, dataloader_test = datasets.utils.get_dataloaders(args.batch_size, args.dataset_name)

    # Get dimensions of the data
    input_dim = next(iter(dataloader_train))[0].shape[1]
    protected_dim = next(iter(dataloader_train))[2].shape[1]
    output_dim = next(iter(dataloader_train))[1].shape[1]

    # Fairness method
    equality_of_odds = True

    # Initialize the predictor based on the dataset
    if args.dataset_name == 'images':
        predictor = ImagePredictor(input_dim, output_dim).to(device)
        equality_of_odds = False
        pytorch_total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {pytorch_total_params}')
    else:
        predictor = Predictor(input_dim).to(device)

    # Initialize the adversary
    adversary = Adversary(input_dim=output_dim, protected_dim=protected_dim, equality_of_odds=equality_of_odds).to(device) if args.debias else None

    # Initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)
    optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr) if args.debias else None

    # Setup the learning rate scheduler
    utils.decayer.step_count = 1
    if args.lr_scheduler == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_P, gamma=0.96) if args.debias else None
    elif args.lr_scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, utils.decayer) if args.debias else None

    # Setup the loss function
    if args.dataset_name == 'crime':
        criterion = nn.MSELoss()
        metric = mean_squared_error
    else:
        criterion = nn.BCELoss()
        metric = accuracy_score

    # Train the model
    train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion,
            metric, adversary, optimizer_A, scheduler, args.alpha, device, args)

    # Test the model
    test(dataloader_test, predictor, adversary, criterion, metric, device, args.dataset_name)
