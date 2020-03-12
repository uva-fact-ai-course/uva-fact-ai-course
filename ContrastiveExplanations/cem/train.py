import torch
import torch.nn as nn

import os.path


def train_ae(
        model,
        dataset,
        iterations=10,
        lr=0.001,
        device='cpu',
        save_fn="mnist-cae",
        load_path="./models/saved_models/mnist-cae.h5"
        ):
    """
    Train autoencoder, or load from save file.

    model
        The autoencoder model to train.
    dataset
        Dataset to train the autoencoder on.
    iterations
        Number of epochs to train the autoencoder for.
    lr
        Initial learning rate.
    device
        Device to train the autoencoder on "cuda" or "cpu".
    save_fn
        Save the trained model to this filename.
    load_path
        Path to load model from, if this file exists and contains a model.
        To override loading weights, specify as empty string.
    """
    model.train()

    if load_path:
        if os.path.isfile(load_path):
            model.load_state_dict(torch.load(load_path, map_location=device))
            model.eval()
            return
        else:
            raise ValueError("invalid load path specified for classifier.")

    # Initialize the device which to run the model on
    device = torch.device(device)

    # specify loss function
    criterion = nn.MSELoss().to(device)

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for j in range(iterations):
        for step, (batch_inputs, _) in enumerate(dataset.train_loader):

            batch_inputs += 0.5

            output = model.forward(batch_inputs.to(device))

            optimizer.zero_grad()

            loss = criterion(output, batch_inputs.to(device))

            loss.backward()
            optimizer.step()

        print("loss after epoch {}:{}".format(j, loss))

        if save_fn:
            torch.save(model.state_dict(),
                       './models/saved_models/' + save_fn + ".h5")

    print('Done training.')
    return


def get_accuracy(predictions, targets):
    """
    Calculates the accuracy for a set of prediction and targets.

    predictions
        Softmax'ed output values of the network.
    targets
        One hot target vectors
    """
    accuracy = (predictions.argmax(1).cpu().numpy() ==
                targets.cpu().numpy()).sum()/(predictions.shape[0])
    return accuracy


def train_cnn(
        model,
        dataset,
        iterations=10,
        lr=0.001,
        batch_size=64,
        device='cpu',
        save_fn="mnist-cnn",
        load_path="./models/saved_models/mnist-cnn.h5"
        ):
    """
    Train CNN, or load from save file.

    model
        The CNN model to train.
    dataset
        Dataset to train the CNN on.
    iterations
        Number of epochs to train the CNN for.
    lr
        Initial learning rate.
    device
        Device to train the CNN on "cuda" or "cpu".
    save_fn
        Save the trained model to this filename.
    load_path
        Path to load model from, if this file exists and contains a model.
        To override loading weights, specify as empty string.
    """

    model.train()

    if load_path:
        if os.path.isfile(load_path):
            model.load_state_dict(torch.load(load_path, map_location=device))
            model.eval()
            return
        else:
            raise ValueError("invalid load path specified for classifier.")

    # Initialize the device which to run the model on
    device = torch.device(device)

    # specify loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Setup the loss and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for j in range(iterations):
        for step, (b_inputs, b_targets) in enumerate(dataset.train_loader):

            output = model.forward(b_inputs.to(device))

            optimizer.zero_grad()

            loss = criterion(output, b_targets.to(device))

            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("loss after step {}:{} accuracy: {}".format(
                    step, loss, get_accuracy(output, b_targets)))

        print("done with iteration: {}/{}".format(j, iterations))

        if save_fn:
            torch.save(model.state_dict(),
                       './models/saved_models/' + save_fn + ".h5")

    print('Done training.')
    return
