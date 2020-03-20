"""
Defines functions to train, test and load models.
"""

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
from src.preprocessing import batch_elastic_transform
from src.model import PrototypeModel, HierarchyModel
from src.helper import check_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_LAMBDA_DICT = {
    'lambda_class' : 20,
    'lambda_class_sup' : 20,
    'lambda_class_sub' : 20,
    'lambda_ae' : 1,
    'lambda_r1' : 1,
    'lambda_r2' : 1,
    'lambda_r3' : 1,
    'lambda_r4' : 1}

def run_epoch(
        evaluation,
        hierarchical, sigma, alpha,                           # Model parameters
        model, dataloader, optimizer,                         # Training objects
        iteration, epoch_loss, epoch_accuracy, sub_accuracy,  # Intermediate results
        lambda_dict):
    """
    Runs through the entire dataset once, updates model only if evaluation=False
    Args:
        Input:
            evaluation : Boolean, if set to true, the model will not be updated
                         and the data will not be warped before the forward pass
            hierarchical : Boolean, is the model hierarchical or not?
            sigma, alpha : Parameters for image warping during training
            model : A PrototypeModel or HierarchyModel
            dataloader : A dataloader object, this function will go through all data
            optimizer : Optimizer object
            iteration, epoch_loss, epoch_accuracy, sub_accuracy : intermediate results
            lambda : all lambda's for calculating the loss function
        Output:
            The output consists of 4 scalars, representing:
            iteration : amount of data points seen
            epoch_loss, epoch_accuracy : accuracy over this epoch
            sub_accuracy : equal to 0 is hierarchical=False
    """

    for _, (images, labels) in enumerate(dataloader):
        # Up the iteration by 1
        iteration += 1

        # Transform images, then port to GPU
        if not evaluation:
            images = batch_elastic_transform(images, sigma, alpha, 28, 28)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        oh_labels = one_hot(labels)

        # Forward pass
        if hierarchical:
            _, decoding, (sub_c, sup_c, r_1, r_2, r_3, r_4) = model.forward(images)
        else:
            _, decoding, (r_1, r_2, c) = model.forward(images)

        # Calculate loss: Crossentropy + Reconstruction + R1 + R2
        # Crossentropy h(f(x)) and y
        ce = nn.CrossEntropyLoss()
        # reconstruction error g(f(x)) and x
        subtr = (decoding - images).view(-1, 28*28)
        re = torch.mean(torch.norm(subtr, dim=1))

        # Paper does 20 * ce and lambda_n = 1 for each regularization term
        # Calculate loss and get accuracy etc.

        if hierarchical:
            sup_ce = ce(sup_c, torch.argmax(oh_labels, dim=1))
            # Extra cross entropy for second linear layer
            sub_ce = ce(sub_c, torch.argmax(oh_labels, dim=1))

            # Actual loss
            loss = lambda_dict['lambda_class_sup'] * sup_ce + \
                lambda_dict['lambda_ae'] * re + \
                lambda_dict['lambda_class_sub'] * sub_ce + \
                lambda_dict['lambda_r1'] * r_1 + \
                lambda_dict['lambda_r2'] * r_2 + \
                lambda_dict['lambda_r3'] * r_3 + \
                lambda_dict['lambda_r4'] * r_4

            # For super prototype cross entropy term
            epoch_loss += loss.item()
            preds = torch.argmax(sup_c, dim=1)
            corr = torch.sum(torch.eq(preds, labels))
            size = labels.shape[0]
            epoch_accuracy += corr.item()/size

            # Also for sub prototype cross entropy term
            subpreds = torch.argmax(sub_c, dim=1)
            subcorr = torch.sum(torch.eq(subpreds, labels))
            sub_accuracy += subcorr.item()/size
        else:
            crossentropy_loss = ce(c, torch.argmax(oh_labels, dim=1))
            loss = lambda_dict['lambda_class'] * crossentropy_loss + \
            lambda_dict['lambda_ae'] * re + \
            lambda_dict['lambda_r1'] * r_1 +  \
            lambda_dict['lambda_r2'] * r_2

            # For prototype cross entropy term
            epoch_loss += loss.item()
            preds = torch.argmax(c, dim=1)
            corr = torch.sum(torch.eq(preds, labels))
            size = labels.shape[0]
            epoch_accuracy += corr.item()/size

        # Do backward pass and ADAM steps
        if not evaluation:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return iteration, epoch_loss, epoch_accuracy, sub_accuracy

def save_images(prototype_path, prototypes, subprototypes, epoch):
    """
    Saves decoded prototypes and decoded subprototypes to the specified folders
    """
    save_image(prototypes, prototype_path+'prot{}.png'.format(epoch), nrow=5, normalize=True)
    if subprototypes is not None:
        save_image(
            subprototypes,
            prototype_path+'subprot{}.png'.format(epoch),
            nrow=5,
            normalize=True)

def test_MNIST(test_data, hierarchical, lambda_dict, results_path, model=None, model_path=None):
    """
    Tests given model on MNIST testdata
    Args:
        Input:
            test_data : data object 
            hierarchical : is the model hierarchical (boolean)
            lambda_dict : lambda's for the test loss 
            results_path : where to save the results
            model/model_path : either a model object or a path to a model (.pt/.pth)
    """
    if model_path is not None:
        model = torch.load(model_path, map_location=torch.device(DEVICE))

    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=250)

    test_loss = 0.0
    test_acc = 0.0
    testsub_acc = 0.0
    it = 0

    it, test_loss, test_acc, testsub_acc = run_epoch(
        True,
        hierarchical,
        None,
        None,
        model,
        test_dataloader,
        None,
        it,
        test_loss,
        test_acc,
        testsub_acc,
        lambda_dict)

    text = "Testdata loss: " +  \
        str(test_loss/it) + \
        " acc: " + \
        str(test_acc/it) + \
        " sub acc: " + \
        str(testsub_acc/it)
    print(text)
    with open(results_path + "results_test.txt", 'w') as f:
        f.write(text)
        f.write('\n')
    return test_loss/it, test_acc/it, testsub_acc/it

def train_MNIST(hierarchical=False, n_prototypes=15, n_sub_prototypes=20,
                latent_size=40, n_classes=10, lambda_dict=DEFAULT_LAMBDA_DICT,
                learning_rate=0.0001, training_epochs=1500,
                batch_size=250, save_every=1, sigma=4, alpha=20, seed=42, directory="my_model",
                underrepresented_class=-1):
    """
    Initializes a new model and trains it on MNIST traindata, then tests it on testdata.
    Args:
        Input:
          Model parameters
            hierarchical : Boolean: Is the model hierarchical?
            n_prototypes : The amount of prototypes. When hierarchical is set to true,
                           this is the amount of superprototypes.
            n_sub_prototypes : The amount of subprototypes. Will be ignored if hierarchical
                               is set to false.
            latent_size : Size of the latent space
            n_classes : Amount of classes
            lambda_dict : Dictionary containing all necessary lambda's for the weighted
                          loss equation
          Training parameters
            learning_rate :
            training_epochs :
            batch_size :
            save_every : how often to save images and models?
          Miscellaneous
            sigma, alpha : Parameters for elastic deformation. Only used for train data
            directory : Directory to save results, prototype images and final model.
            underrepresented model : The class that is to be downsampled (0.25 to 1 for all
                                     other classes) When it is set to -1, no class is downsampled.
    """
    # Default settings for hierarchical model
    if hierarchical:
        n_prototypes = 10

    # Set torch seed
    torch.manual_seed(seed)

    # Prepare directories
    check_path('results')
    directory = 'results/' + directory
    check_path(directory)

    model_path = directory + '/models/'
    prototype_path = directory + '/prototypes/'
    decoding_path = directory + '/decoding/'
    results_path = directory +'/results/'

    check_path(model_path)
    check_path(prototype_path)
    check_path(decoding_path)
    check_path(results_path)

    # Prepare files
    f = open(results_path + "results_s" + str(seed) + ".txt", "w")
    f.write(', '.join([str(x) for x in [hierarchical, n_prototypes, n_sub_prototypes,
                                        latent_size, learning_rate]]))
    f.write('\n')
    f.close()

    ### Load data
    train_data = MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_data = MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    ### Initialize the model and the optimizer.
    if hierarchical:
        proto = HierarchyModel(n_prototypes, latent_size, n_classes, n_sub_prototypes)
    else:
        proto = PrototypeModel(n_prototypes, latent_size, n_classes)

    proto = proto.to(DEVICE)
    optim = torch.optim.Adam(proto.parameters(), lr=learning_rate)

    ### Initialize the dataloader
    if underrepresented_class > -1:
        labels = [label for _, label in train_data]
        train_samples_weight = torch.tensor([0.25 if class_id == underrepresented_class
                                             else 1 for class_id in labels]).double()
        dataloader = DataLoader(train_data, batch_size=batch_size, sampler=WeightedRandomSampler(
            weights=train_samples_weight,
            num_samples=len(labels)))
        print("Got dataloader loaded with WeightedRandomSampler")
    else:
        dataloader = DataLoader(train_data, batch_size=batch_size)

    ### Run for a specified number of epochs
    for epoch in range(training_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        sub_acc = 0.0
        it = 0

        it, epoch_loss, epoch_acc, sub_acc = run_epoch(
            False,
            hierarchical,
            sigma,
            alpha,
            proto,
            dataloader,
            optim,
            it,
            epoch_loss,
            epoch_acc,
            sub_acc,
            lambda_dict)

        # To save time
        if epoch % save_every == 0:
        # Get prototypes and decode them to display
            prototypes = proto.prototype.get_prototypes()
            prototypes = prototypes.view(-1, 10, 2, 2)
            imgs = proto.decoder(prototypes)

            subprototypes = None
            if hierarchical:
                subprototypes = proto.prototype.get_sub_prototypes()
                for i in range(len(subprototypes)):
                    if i == 0:
                        subprotoset = subprototypes[i].view(-1, 10, 2, 2)
                    else:
                        subprotoset = torch.cat([subprotoset, subprototypes[i].view(-1, 10, 2, 2)])
                subprototypes = proto.decoder(subprotoset)

            # Save images
            save_images(prototype_path, imgs, subprototypes, epoch)

            # Save model
            torch.save(proto, model_path+"proto{}.pth".format(seed))

        # Print statement to check on progress
        with open(results_path + "results_s" + str(seed) + ".txt", "a") as f:
            text = "Epoch: " + \
                str(epoch) + \
                " loss: " + \
                str(epoch_loss / it) + \
                " accuracy: " + \
                str(epoch_acc/it)
            text += " sub_accuracy: " + str(sub_acc/it) if hierarchical else ""
            print(text)
            f.write(text)
            f.write('\n')

    ### Save final model, no matter what
    torch.save(proto, model_path+"final.pth")

    ### Test model on testdata
    t_loss, t_acc, t_sub = test_MNIST(
        test_data,
        hierarchical,
        lambda_dict,
        results_path,
        model=proto)
    return t_loss, t_acc, t_sub

def load_and_test(path, hierarchical):
    """
    Input: path to a model and if the model is hierarchical
    loads and tests the model for accuracy and loss
    """
    test_data = MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    test_MNIST(test_data, hierarchical, DEFAULT_LAMBDA_DICT, '', model_path=path)
