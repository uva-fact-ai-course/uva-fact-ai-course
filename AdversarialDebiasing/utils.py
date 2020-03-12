import torch

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from collections import Counter
from collections import defaultdict


def decayer(lr):
    """
    Return a decayed learning rate based on the step counter

    Args:
        lr (float): Learning rate

    Returns: New learning rate which is decayed

    """
    new_lr = lr / decayer.step_count
    return new_lr


def forward_full(dataloader, predictor, adversary, criterion, device, dataset_name, optimizer_P=None, optimizer_A=None, scheduler=None, train=False, alpha=0.3):
    """
    Performs one epoch of training/evaluation on the data

    Args:
        dataloader (DataLoader): Dataloader for the data
        predictor (nn.Module): Predictor model
        adversary (nn.Module): Adversary model
        criterion (func): Loss criterion
        device (torch.device): Device on which to train/evaluate
        dataset_name (str): Name of the dataset
        optimizer_P (Optimizer): Optimizer for the predictor
        optimizer_A (Optimizer): Optimizer for the adversary
        scheduler (func): Learning rate scheduler
        train (bool): True for training mode, False for evaluation
        alpha (float): Value of hyperparameter alpha

    Returns: Metrics from training/evaluation

    """
    labels_dict = {'true': [], 'pred': []}
    protected_dict = {'true': [], 'pred': []}
    losses_P, losses_A = [], []

    for i, (x, y, z) in enumerate(dataloader):

        x = x.to(device)
        true_y_label = y.to(device)
        true_z_label = z.to(device)

        # Forward step through predictor
        pred_y_logit, pred_y_prob = predictor(x)

        if train is False:
            if i == 0:
                prediction_probs = pred_y_prob.cpu().detach().numpy()
            else:
                prediction_probs = np.concatenate((prediction_probs, pred_y_prob.cpu().detach().numpy()), axis=0)

        # Compute loss with respect to predictor
        loss_P = criterion(pred_y_prob, true_y_label)
        losses_P.append(loss_P.item())

        # Store the true labels and the predictions
        if dataset_name == 'images':
            labels_dict['true'].extend(torch.max(true_y_label, dim=1)[1].cpu().numpy().tolist())
            labels_dict['pred'].extend(torch.max(pred_y_prob, dim=1)[1].cpu().numpy().tolist())
        elif dataset_name == 'adult':
            labels_dict['true'].extend(y.squeeze().cpu().numpy().tolist())
            pred_y = (pred_y_prob > 0.5).int().squeeze(dim=1).cpu().numpy().tolist()
            labels_dict['pred'].extend(pred_y)
        else:
            labels_dict['true'].extend(y.cpu().numpy().tolist())
            labels_dict['pred'].extend(pred_y_prob.detach().cpu().numpy().tolist())
        protected_dict['true'].extend(z.squeeze().cpu().numpy().tolist())

        if adversary is not None:
            # Forward step through adversary
            pred_z_logit, pred_z_prob = adversary(pred_y_logit, true_y_label)

            # Compute loss with respect to adversary
            loss_A = criterion(pred_z_prob, true_z_label)
            losses_A.append(loss_A.item())

            if dataset_name == 'crime':
                pred_z = pred_z_prob.detach().numpy().tolist()
            else:
                pred_z = (pred_z_prob > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
            protected_dict['pred'].extend(pred_z)

        if train:
            if adversary is not None:
                # Reset gradients of adversary and predictor
                optimizer_A.zero_grad()
                optimizer_P.zero_grad()
                # Compute gradients of adversary loss
                loss_A.backward(retain_graph=True)
                # Concatenate gradients of adversary loss with respect to the predictor
                grad_w_La = concat_grad(predictor)

            # Reset gradients of predictor
            optimizer_P.zero_grad()

            # Compute gradients of predictor loss
            loss_P.backward()

            if adversary is not None:
                # Concatenate gradients of predictor loss with respect to the predictor
                grad_w_Lp = concat_grad(predictor)
                # Project gradients of the predictor
                proj_grad = project_grad(grad_w_Lp, grad_w_La)
                # Modify and replace the gradient of the predictor
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                replace_grad(predictor, grad_w_Lp)

            # Update predictor weights
            optimizer_P.step()
            
            if adversary is not None:
                # Decay the learning rate
                decayer.step_count += 1
                if decayer.step_count % 1000 == 0:
                    scheduler.step()

                # Update adversary weights
                optimizer_A.step()

    if train:
        return losses_P, losses_A, labels_dict, protected_dict
    else:
        return losses_P, losses_A, labels_dict, protected_dict, prediction_probs


def concat_grad(model):
    """
    Concatenates the gradients of a model to form a single parameter vector tensor.

    Args:
        model (nn.Module): PyTorch model object

    Returns: A single vector tensor with the model gradients concatenated
    """
    g = None
    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=0)
        if g is None:
            g = param.grad.view(1, -1)
        else:
            if len(grad.shape) < 2:
                grad = grad.unsqueeze(dim=0)
            else:
                grad = grad.view(1, -1)
            g = torch.cat((g, grad), dim=1)
    return g.squeeze(dim=0)


def replace_grad(model, grad):
    """
    Replaces the gradients of the model with the specified gradient vector tensor.

    Args:
        model (nn.Module): PyTorch model object
        grad (Tensor): Vector of concatenated gradients

    Returns: None
    """
    start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param.grad.data = grad[start:start + numel].view(param.grad.shape)
        start += numel


def project_grad(x, v):
    """
    Performs a projection of one vector on another.

    Args:
        x (Tensor): Vector to project
        v (Tensor): Vector on which projection is required

    Returns: Tensor containing the projected vector
    """
    norm_v = v / (torch.norm(v) + torch.finfo(torch.float32).tiny)
    proj_grad = torch.dot(x, norm_v) * v
    return proj_grad


def calculate_fpr(fp, tn):
    """
    Calculates false positive rate (FPR).

    Args:
        fp (int): Number of false positives
        tn (int): Number of true negatives

    Returns: False positive rate
    """
    return fp / (fp + tn)


def calculate_fnr(fn, tp):
    """
    Calculates false negative rate (FNR).

    Args:
        fn (int): Number of false negatives
        tp (int): Number of true positives

    Returns: False negative rate
    """
    return fn / (fn + tp)


def calculate_metrics(true_labels, predictions, true_protected, dataset, pred_probs=None):
    """
    Calculate metrics for reporting

    Args:
        true_labels (list): True labels
        predictions (list): Predictions from the model
        true_protected (list): True values of the protected variable
        dataset (str): Name of the dataset
        pred_probs (list): Prediction probability estimates

    Returns:
        Set of FPR, FNR metrics and confusion matrix for UCI Adult dataset
        OR
        Set of precision, recall, F1, support, AUC metrics and difference in conditional probabilities for UTKFace experiment
    """

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    negative_indices = np.where(np.array(true_protected) == 0)[0]
    positive_indices = np.where(np.array(true_protected) == 1)[0]
    neg_confusion_mat = confusion_matrix(true_labels[negative_indices], predictions[negative_indices])
    pos_confusion_mat = confusion_matrix(true_labels[positive_indices], predictions[positive_indices])

    if dataset == 'adult':
        tn, fp, fn, tp = neg_confusion_mat.ravel()
        neg_fpr = calculate_fpr(fp, tn)
        neg_fnr = calculate_fnr(fn, tp)

        tn, fp, fn, tp = pos_confusion_mat.ravel()
        pos_fpr = calculate_fpr(fp, tn)
        pos_fnr = calculate_fnr(fn, tp)

        return neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr

    elif dataset == 'images':
        # 0 is male, so negative = male; positive = female
        neg_conditionals = conditional_matrix(neg_confusion_mat)
        pos_conditionals = conditional_matrix(pos_confusion_mat)
        protected_differences = neg_conditionals - pos_conditionals

        # Difference between conditionals measures the degree to which equality of odds is satisfied
        avg_dif = np.average(protected_differences, axis=1)
        avg_abs_dif = np.average(np.absolute(protected_differences), axis=1)

        neg_prec, neg_recall, neg_fscore, neg_support = precision_recall_fscore_support(true_labels[negative_indices],
                                                                                        predictions[negative_indices])
        pos_prec, pos_recall, pos_fscore, pos_support = precision_recall_fscore_support(true_labels[positive_indices],
                                                                                        predictions[positive_indices])
        if pred_probs is not None:
            one_hot_labels = np.zeros((true_labels.size, true_labels.max()+1))
            one_hot_labels[np.arange(true_labels.size),true_labels] = 1
            neg_auc = roc_auc_score(one_hot_labels[negative_indices], pred_probs[negative_indices])
            pos_auc = roc_auc_score(one_hot_labels[positive_indices], pred_probs[positive_indices])

        return neg_prec, neg_recall, neg_fscore, neg_support, neg_auc, pos_prec, pos_recall, pos_fscore, pos_support, pos_auc, avg_dif, avg_abs_dif


def conditional_matrix(confusion_matrix):
    """
    Computes a matrix of conditional probabilities based on the following formula:
    p(y_hat| y) = p(y_hat, y) / p(y)
    
    Args:
        confusion_matrix(np.array), where y axis = true label, x axis = pred label

    Returns: conditional_matrix (np.array)
    """
    normalization = np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)
    conditional_matrix = np.divide(confusion_matrix, normalization)
    return conditional_matrix


def plot_learning_curves(P, A, dataset='adult'):
    """
    Plots the learning curves

    Args:
        P (list of lists): Contains the train losses, train scores, validation losses and validation scores for the predictor
        A (list of lists): Contains the train losses, train scores, validation losses and validation scores for the adversary
        dataset (str): Name of the dataset

    Returns: None

    """
    if dataset == 'crime':
        metric = 'MSE'
    else:
        metric = 'accuracy'

    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(1, len(P[0])+1), P[0], label="Train loss predictor", color="#E74C3C")
    axs[0].plot(np.arange(1, len(P[2])+1), P[2], label="Val loss predictor", color="#8E44AD")

    axs[2].plot(np.arange(1, len(P[1])+1), P[1], label=str('Train %s predictor'%(metric)), color="#229954")
    axs[2].plot(np.arange(1, len(P[3])+1), P[3], label=str('Val %s predictor'%(metric)), color="#E67E22")

    axs[1].plot(np.arange(1, len(A[0])+1), A[0], label="Train loss adversary", color="#3498DB")
    axs[1].plot(np.arange(1, len(A[2])+1), A[2], label="Val loss adversary", color="#FFC300")

    axs[3].plot(np.arange(1, len(A[1])+1), A[1], label=str('Train %s adversary'%(metric)), color="#229954")
    axs[3].plot(np.arange(1, len(A[3])+1), A[3], label=str('Val %s adversary'%(metric)), color="#E67E22")

    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="upper right")

    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MSE') if dataset == 'crime' else axs[2].set_ylabel('Accuracy')
    axs[2].legend(loc="upper right")

    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('MSE') if dataset == 'crime' else axs[3].set_ylabel('Accuracy')
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    if A[0] == []:
        title = f'train_{dataset}_no_debias.png'
    else:
        title = f'train_{dataset}_debias.png'
    plt.savefig(title)


def make_coplot(protected_dict, labels_dict):
    """
    Produces a conditional plot in which predictions are conditioned on the
    targets of the data samples. The samples are grouped into five consecutive
    bins, based on their values for the protected variable.

    Args:
        protected_dict (dict): Predicted and true values of protected variable
        labels_dict (dict): Predicted and true labels

    Returns: None
    """
    plt.figure()

    ratios = np.array(protected_dict['true'])
    true_crime_rates = np.array(labels_dict['true'])
    pred_crime_rates = np.array(labels_dict['pred'])
    bin_width = 0.2
    for l_edge in np.arange(0, 1, bin_width):
        # Bin samples based on ratio of white people in the community
        r_edge = l_edge + bin_width
        bin = str('%.1f < z <= %.1f' %(l_edge, r_edge))

        idcs = np.where((l_edge < ratios) & (ratios <= r_edge))[0]
        # Sort samples within bin based on their true targets,
        # for clean visualisation
        s = sorted(zip(true_crime_rates[idcs], pred_crime_rates[idcs]))
        x, y = map(np.array, zip(*s))

        # Plot datapoins in bin, and (linearly) fit line to them
        points = plt.plot(x, y, '.', alpha=0.42, label=bin)
        a, b = np.polyfit(x.flatten(), y.flatten(), 1)
        fitted_line = [a*i + b for i in x]
        plt.plot(x, fitted_line, color = points[0].get_color())

    plt.plot([0,1], [0,1], 'k--', label='perfect predictor')
    plt.xlabel('True crime rate')
    plt.ylabel('Predicted crime rate')
    plt.title('Predicted and true crime rates for different values of z,\n\
               the ratio of white people within the community')
    plt.legend(loc='lower right', ncol=1)
    plt.show()


def plot_adult_results(neg_fnr_b, pos_fnr_b, neg_fpr_b, pos_fpr_b, neg_fnr_db, pos_fnr_db, neg_fpr_db, pos_fpr_db):
    """ 
    Plots FNR and FPR for male and female in the biased and debiased setting

    Args:
        neg_fnr_b (list): List of fnr for female in the biased setting
        pos_fnr_b (list): List of fnr for male in the biased setting
        neg_fpr_b (list): List of fpr for female in the biased setting
        pos_fpr_b (list): List of fpr for male in the biased setting
        neg_fnr_db (list): List of fnr for female in the debiased setting
        pos_fnr_db (list): List of fnr for male in the debiased setting
        neg_fpr_db (list): List of fpr for female in the debiased setting
        pos_fpr_db (list): List of fpr for male in the debiased setting

    Returns: None
    """
    
    # Layout
    plt.style.use('seaborn-whitegrid')
    new_style = {'grid': False}
    plt.rc('axes', **new_style)
    
    # Set distances
    bar_width = 0.25
    r1 = [0, 0.7]
    r2 = [x + bar_width for x in r1]

    fig, axs = plt.subplots(1, 2, figsize=(7, 4))

    # Plot results FNR
    women_mean_fnr = (np.mean(neg_fnr_b), np.mean(neg_fnr_db))
    men_mean_fnr = (np.mean(pos_fnr_b), np.mean(pos_fnr_db))
    women_std_fnr = (np.std(neg_fnr_b), np.std(neg_fnr_db))
    men_std_fnr = (np.std(pos_fnr_b), np.std(pos_fnr_db))

    # Plot standard deviations if multiple results are available
    if len(neg_fnr_b) > 1:
        axs[0].bar(r1, women_mean_fnr, color='#FF00B2', width=bar_width, yerr=women_std_fnr, 
                    error_kw=dict(lw=0.7, capsize=4, capthick=0.7, ecolor="#505050"), 
                    edgecolor='white', label='Women')
        axs[0].bar(r2, men_mean_fnr, color='#0097FF', width=bar_width, yerr=men_std_fnr, 
                    error_kw=dict(lw=0.7, capsize=4, capthick=0.7, ecolor="#505050"), 
                    edgecolor='white', label='Men')
    else:
        axs[0].bar(r1, women_mean_fnr, color='#FF00B2', width=bar_width, edgecolor='white', label='Female')
        axs[0].bar(r2, men_mean_fnr, color='#0097FF', width=bar_width, edgecolor='white', label='Male')  
        
        
    
    # Plot FNR results from Zhang et al
    axs[0].plot([r1[0]-0.5*bar_width +0.005, r1[0]+0.5*bar_width], [0.4492, 0.4492], "k:", lw=2, label="Zhang et al. (2018)")
    axs[0].plot([r1[1]-0.5*bar_width +0.005, r1[1]+0.5*bar_width], [0.4458, 0.4458], "k:", lw=2)
    axs[0].plot([r2[0]-0.5*bar_width +0.005, r2[0]+0.5*bar_width], [0.3667, 0.3667], "k:", lw=2)
    axs[0].plot([r2[1]-0.5*bar_width +0.005, r2[1]+0.5*bar_width], [0.4349, 0.4349], "k:", lw=2)

    axs[0].set_xlabel('FNR', fontsize=12)
    axs[0].set_xticks([r + bar_width/2 for r in r1])
    axs[0].set_xticklabels(['Without debias', 'With debias'], fontsize=12)
    axs[0].set_ylabel('Rate', fontsize=12)

    # Plot results FPR
    women_mean_fpr = (np.mean(neg_fpr_b), np.mean(neg_fpr_db))
    men_mean_fpr = (np.mean(pos_fpr_b), np.mean(pos_fpr_db))
    women_std_fpr = (np.std(neg_fpr_b), np.std(neg_fpr_db))
    men_std_fpr = (np.std(pos_fpr_b), np.std(pos_fpr_db))

    # Plot standard deviations if multiple results are available
    if len(neg_fnr_b) > 1: 
        axs[1].bar(r1, women_mean_fpr, color='#FF00B2', width=bar_width, yerr=women_std_fpr, 
                    error_kw=dict(lw=0.7, capsize=4, capthick=0.7, ecolor="#505050"), 
                    edgecolor='white', label='Female')
        axs[1].bar(r2, men_mean_fpr, color='#0097FF', width=bar_width, yerr=men_std_fpr, 
                    error_kw=dict(lw=0.7, capsize=4, capthick=0.7, ecolor="#505050"), 
                    edgecolor='white', label='Male')
    else:
        axs[1].bar(r1, women_mean_fpr, color='#FF00B2', width=bar_width, edgecolor='white', label='Female')
        axs[1].bar(r2, men_mean_fpr, color='#0097FF', width=bar_width, yerr=men_std_fpr, edgecolor='white', label='Male')

    # Plot FPR results from Zhang et al
    axs[1].plot([r1[0]-0.5*bar_width +0.005, r1[0]+0.5*bar_width], [0.0248, 0.0248],"k:", lw=2, label="Zhang et al.")
    axs[1].plot([r1[1]-0.5*bar_width +0.005, r1[1]+0.5*bar_width], [0.0647, 0.0647], "k:", lw=2)
    axs[1].plot([r2[0]-0.5*bar_width +0.005, r2[0]+0.5*bar_width], [0.0917, 0.0917], "k:", lw=2)
    axs[1].plot([r2[1]-0.5*bar_width +0.005, r2[1]+0.5*bar_width], [0.0701, 0.0701], "k:", lw=2)

    axs[1].set_xlabel('FPR', fontsize=12)
    axs[1].set_xticks([r + bar_width/2 for r in r1])
    axs[1].set_xticklabels(['Without debias', 'With debias'], fontsize=12)
    axs[1].legend(loc="upper right", fontsize=12)
    plt.tight_layout()


def entropy(rv1, cond_rv=None):
    """
    Calculates either the entropy or conditional entropy depending on if a
    conditional random variable is given.

    Args:
        rv1 (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable rv1
        cond_rv (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable cond_rv

    Returns: Entropy or conditional entropy
    """

    entropy = 0

    if cond_rv is None:
        # Calculate entropy H(rv1)
        distr_rv1 = get_distr(rv1)
        for prob in distr_rv1.values():
            entropy += prob * math.log(1 / prob, 2)
    else:
        # Calculate entropy H(rv1 | cond_rv)
        conditional_distr_rv1 = get_conditional_distr(rv1, cond_rv)
        distr_cond_rv = get_distr(cond_rv)

        for event, prob in distr_cond_rv.items():
            entropy_part = 0
            for cond_prob in conditional_distr_rv1[event].values():
                entropy_part += cond_prob * math.log(1 / cond_prob, 2)
            entropy += prob * entropy_part
    return entropy


def get_joint(rv1, rv2):
    """
    Gets all pairs of samples from the random variables that occur together.

    Args:
        rv1 (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable rv1
        rv2 (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable rv2

    Returns: List of tuples of all samples in rv1 and rv2 occurring together
    """
    return [i for i in zip(rv1, rv2)]


def get_distr(rv):
    """
    Calculates probabilities of the random variable rv.

    Args:
        rv (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable rv

    Returns: Dictionary of probabilities where the keys are the events of rv
    """
    distr = {}
    for event, frequency in Counter(rv).items():
        distr[event] = frequency / len(rv)
    return distr


def get_conditional_distr(rv, cond_rv):
    """
    Calculates conditional probabilties of the random variable rv
    given another random variable cond_rv.

    Args:
        rv (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable rv
        cond_rv (list): List where every element i corresponds to the mapping of outcome i 
                into the image of random variable cond_rv

    Returns: Dictionary of conditional probabilities where the first keys
        are events of the conditional random variable and the second keys are
        events of the random variable
    """

    # Get joint distribution
    distr_joint = get_distr(get_joint(rv, cond_rv))

    # Get distribution of random variable we want to condition on
    distr_cond_rv = get_distr(cond_rv)

    # Get distribution of random variable given conditional random variable by Bayes' rule
    conditional_distr = defaultdict(lambda: {})
    for event, joint_prob in distr_joint.items():
        conditional_distr[event[1]][event[0]] = joint_prob / distr_cond_rv[event[1]]

    return conditional_distr


def mutual_information(rv1, rv2, cond_rv=None):
    """
    Calculates mutual information between random variables rv1 and rv2 eventually
    conditioned on another random variable cond_rv if given.

    Args:
        rv1 (list): List where every element i corresponds to the mapping of outcome i 
            into the image of random variable rv1
        rv2 (list): List where every element i corresponds to the mapping of outcome i 
            into the image of random variable rv2
        cond_rv (list): List where every element i corresponds to the mapping of outcome i 
            into the image of random variable cond_rv

    Returns: Mutual information
    """

    mutual_information = None

    if cond_rv is None:

        # Compute entropy H(rv1)
        entropy1 = entropy(rv1)

        # Compute entropy H(rv1 | rv2)
        entropy2 = entropy(rv1, cond_rv = rv2)

        # Compute mutual information I(rv1; rv2)
        mutual_information = entropy1 - entropy2

    else:
        # Compute entropy H(rv1 | cond_rv)
        entropy1 = entropy(rv1, cond_rv = cond_rv)

        # Compute entropy H(rv1 | cond_rv, rv2)
        entropy2 = entropy(rv1, cond_rv = get_joint(cond_rv, rv2))

        # # Compute mutual information I(rv1; rv2 | cond_rv)
        mutual_information = entropy1 - entropy2

    return mutual_information
