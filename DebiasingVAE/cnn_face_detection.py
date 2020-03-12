import torch
import numpy as np
from torch import nn
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# Setting a plot style.
sns.set(style="darkgrid")


class CNN(nn.Module):
    """
    This class holds the standard CNN face detection classifier.
    """

    def __init__(self):
        """
        During initialization we define the two main components of the system. The convolution stack and the top
        classifier. The architecture is choosen to represent the one in the paper. Number of filters are chosen
        as in the affiliated MIT implementation.
        """
        super().__init__()

        # Check if CUDA support is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # The number of filters used in respective convolution layers.
        self.n_filters = {
            'conv_1': 3,
            'conv_2': 12,
            'conv_3': 24,
            'conv_4': 48,
            'conv_5': 72
        }

        # The used kernel size.
        self.kernel_size = {
            'conv_1': 5,
            'conv_2': 5,
            'conv_3': 5,
            'conv_4': 5,
        }

        # The used kernel size.
        self.stride = {
            'conv_1': 2,
            'conv_2': 2,
            'conv_3': 2,
            'conv_4': 2,
        }

        # This shape is the one at the end of the conv_block.
        self.pre_flatten_shape = (None, 72, 1, 1)

        # The flattened output-shape of the CNN.
        flatten_cnn_out_shape = np.prod(self.pre_flatten_shape[1:])

        # Number of outputs on classifier (here a single value showing probability of face-class)
        self.n_outputs = 1
        # The dimensions of the hidden layer of the classifier.
        self.n_hidden = {
            'hidden_1': flatten_cnn_out_shape,
            'hidden_2': 1000,
        }

        # The convolution block.
        self.conv_stack = nn.Sequential(

            # Conv block 1.
            nn.Conv2d(self.n_filters['conv_1'], self.n_filters['conv_2'],
                      kernel_size=self.kernel_size['conv_1'], stride=self.stride['conv_1']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['conv_2']),

            # Conv block 2.
            nn.Conv2d(self.n_filters['conv_2'], self.n_filters['conv_3'],
                      kernel_size=self.kernel_size['conv_2'], stride=self.stride['conv_2']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['conv_3']),

            # Conv block 3.
            nn.Conv2d(self.n_filters['conv_3'], self.n_filters['conv_4'],
                      kernel_size=self.kernel_size['conv_3'], stride=self.stride['conv_3']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['conv_4']),

            # Conv block 4.
            nn.Conv2d(self.n_filters['conv_4'], self.n_filters['conv_5'],
                      kernel_size=self.kernel_size['conv_4'], stride=self.stride['conv_4']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['conv_5']),
        )

        # The classifier on top of the convolution block to get the class probabilities.
        self.classifier = nn.Sequential(

            # Hidden layer.
            nn.Linear(self.n_hidden['hidden_1'], self.n_hidden['hidden_2']),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden['hidden_2']),

            # Output header
            nn.Linear(self.n_hidden['hidden_2'], self.n_outputs),
            nn.Sigmoid(),
        )

        # A placeholder for metric plots.
        self.metric_plots = dict()

    def forward(self, x):
        """
        This is the forward method the is called to get the output of the network given the input x.
        Args:
            x: The image input of the network in batch from.

        Returns: The activation of the final network layer, representing the face class probability.

        """

        # Getting the activation of the convolution block.
        out = self.conv_stack(x)

        # make sure the variable is properly set
        assertion_text = "The defined pre-flatten shape is incorrect," \
                         " got {} but expected {}.".format(out.shape, self.pre_flatten_shape)
        assert (out.shape[1:] == self.pre_flatten_shape[1:]), assertion_text

        # Flatten the output of the convolution to allow for input in fully-connected header.
        out = out.flatten(start_dim=1)

        # Get the class probability.
        pred_probability = self.classifier(out)

        return pred_probability

    def run_training(self, batch_size, num_epochs, train_dataset, title='CNN Training', validation_dataset=None):
        """
        The training function of the CNN. If a validation dataset is passed the validation epoch is evenly
        split across the training epoch. So that both epoch end at the same time.
        Args:
            batch_size: Batch size to be used in training and validation.
            num_epochs: Number of epochs to train.
            train_dataset: The dataset loader for trainingset.
            title: The title of the training stats figure.
            validation_dataset: If given the loader of the validation dataset.

        Returns: A dict holding, the training and validation loss and accuracy lists over iterations
        """

        # Set the model in training mode.
        self.train()

        # The learning rate to be used in the optimizer.
        learning_rate = 1e-4
        # Setup the Binary Cross Entropy loss.
        criterion = torch.nn.BCELoss().to(self.device)
        # Set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)

        # Init of the metric plots that show the training progress.
        self.metric_plots['fig'] = plt.figure(figsize=(9, 8), num=title)
        self.metric_plots['plt1'] = self.metric_plots['fig'].add_subplot(2, 1, 1)
        self.metric_plots['plt1_legend'] = None
        self.metric_plots['plt2'] = self.metric_plots['fig'].add_subplot(2, 1, 2)
        self.metric_plots['plt2_legend'] = None
        self.metric_plots['fig'].suptitle(title, y=0.93)

        # Init a train loader that shuffles the data and returns batches of size batch_size.
        # The last batch is dropped so that we have no problems with the batch norm.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # We calculate the numbers of iterations in each epochs.
        num_train_iters_per_epoch = len(train_dataset) // batch_size
        # Then we can get the overall number of iterations in the training process, which is used to measure progress.
        num_train_iters = num_train_iters_per_epoch * num_epochs

        # In case we passed a validation dataset we also need to prepare it's usage.
        if validation_dataset:
            # Define a validation data loader similar as for training above.
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # Calculate the iterations per epoch of validation.
            num_val_iters_per_epoch = len(validation_dataset) // batch_size

            # Then we space the evaluation iterations evenly across a training epoch, so that they finish together.
            val_iters_2_train_iters = np.linspace(start=0, stop=num_train_iters_per_epoch,
                                                  num=num_val_iters_per_epoch,
                                                  endpoint=False,
                                                  dtype=int)

        # In case there is no validation dataset passed, we just define no validation iters and a dummy val loader.
        else:
            val_iters_2_train_iters = []
            val_loader = None

        # Theses lists hold the loss and accuracy metrics over training iters.
        iter_val_loss_list = list()
        iter_val_acc_list = list()
        iter_train_loss_list = list()
        iter_train_acc_list = list()

        # We initialize a progressbar over total number of iterations for training.
        pbar = tqdm(total=num_train_iters, position=0, unit="it", leave=True)

        # The main training loop starts here:
        for epoch in range(num_epochs):

            # In case validation dataset is passed we initialize a iteration function over it's content.
            if validation_dataset:
                val_loader_iter = iter(val_loader)

            # For all batches in the train loader.
            for iter_id, sample_batched in enumerate(train_loader):

                # Unpack batch of training data and convert the input images and labels to tensors
                image_batch, image_labels = sample_batched
                image_batch = torch.as_tensor(image_batch, dtype=torch.float32).to(self.device)
                image_labels = torch.as_tensor(image_labels, dtype=torch.float32).to(self.device)
                # Permute the image channels so that they are as expected in pytorch (channels x h x w).
                image_batch = image_batch.permute(0, 3, 2, 1)

                # Assure that the first dimension of the batch equals the set batch_size.
                assert image_batch.shape[0] == batch_size

                # Call the model with the image data to get the prediction probabilities.
                face_prob = self.forward(image_batch)

                # Calculate the loss of the model based on predictions and labels.
                loss = criterion(face_prob, image_labels.view(-1, 1))

                # Reset the gradients in the network.
                optimizer.zero_grad()
                # Backprob the loss through the network.
                loss.backward()
                # Make a step of optimization.
                optimizer.step()

                # Calculate the accuracy in this iteration.
                iter_train_acc = self.calculate_accuracy(labels=image_labels, predictions=face_prob)

                # Calculate the global iteration counter used for plotting.
                global_iter = epoch * num_train_iters_per_epoch + iter_id

                # Save the training acc and loss to the respective lists.
                iter_train_loss_list.append([global_iter, loss.item()])
                iter_train_acc_list.append([global_iter, iter_train_acc])

                # In case of validation and if the training iteration is also a validation iteration.
                if validation_dataset and iter_id in val_iters_2_train_iters:
                    # Set the model in validation mode.
                    self.eval()

                    # Retreive a batch of validation data.
                    image_batch, image_labels = next(val_loader_iter, (None, None))
                    # Assert there is data in the batch.
                    assert image_batch is not None

                    # Cast the images and labesl to tensors.
                    image_batch = torch.as_tensor(image_batch, dtype=torch.float32).to(self.device)
                    image_labels = torch.as_tensor(image_labels, dtype=torch.float32).to(self.device)
                    # Permute the image channels so that they are as expected in pytorch (channels x h x w).
                    image_batch = image_batch.permute(0, 3, 2, 1)

                    # Without using gradient related computations.
                    with torch.no_grad():
                        # Assert batch size is as expected.
                        assert image_batch.shape[0] == batch_size

                        # Call the model with the image data to get the prediction probabilities.
                        face_prob = self.forward(image_batch)

                        # Calculate the loss of the model based on predictions and labels.
                        loss = criterion(face_prob, image_labels.view(-1, 1))

                        # Calculate the accuracy in this iteration.
                        iter_val_acc = self.calculate_accuracy(labels=image_labels, predictions=face_prob)

                    # Save the validation acc and loss to the respective lists.
                    iter_val_loss_list.append([global_iter, loss.item()])
                    iter_val_acc_list.append([global_iter, iter_val_acc])

                    # Set the model in training mode again.
                    self.train()

                # Every 200 iterations we redraw the training metric plots.
                if iter_id % 200 == 0:
                    self.update_plots(
                        iter_train_loss_list=iter_train_loss_list,
                        iter_train_acc_list=iter_train_acc_list,
                        iter_val_loss_list=iter_val_loss_list,
                        iter_val_acc_list=iter_val_acc_list,
                    )

                # We update the overall training progress of the model.
                pbar.update()

            # Calculate the mean training loss and acc on the epoch for print statements.
            mean_train_acc_epoch = np.asanyarray(iter_train_acc_list[-num_train_iters_per_epoch:])[:, 1].mean()
            mean_train_loss_epoch = np.asanyarray(iter_train_loss_list[-num_train_iters_per_epoch:])[:, 1].mean()

            # In case of validation.
            if validation_dataset:
                # Calculate the mean validation loss and acc on the epoch for print statements.
                mean_val_acc_epoch = np.asanyarray(iter_val_acc_list[-num_val_iters_per_epoch:])[:, 1].mean()
                mean_val_loss_epoch = np.asanyarray(iter_val_loss_list[-num_val_iters_per_epoch:])[:, 1].mean()

                # Print epoch stats including validation.
                print(
                    "Epoch {:0>2d}/{:0>2d}: mean acc train/val={:.2%}/{:.2%} ;"
                    " mean train/val loss={:.2e}/{:.2e}".format(
                        epoch + 1,
                        num_epochs,
                        mean_train_acc_epoch,
                        mean_val_acc_epoch,
                        mean_train_loss_epoch,
                        mean_val_loss_epoch,
                    )
                )

            else:
                # Print epoch stats.
                print(
                    "Epoch {:0>2d}/{:0>2d}: mean acc train/val={:.2%}/-- ;"
                    " mean train/val loss={:.2e}/--".format(
                        epoch + 1, num_epochs, mean_train_acc_epoch, mean_train_loss_epoch,
                    )
                )

        # Close the progress bar at the end of the training
        pbar.close()

        # Create the training/validation stats dictionary.
        stats = {
            'iter_train_loss_list': iter_train_loss_list,
            'iter_train_acc_list': iter_train_acc_list,
            'iter_val_loss_list': iter_val_loss_list,
            'iter_val_acc_list': iter_val_acc_list
        }

        return stats

    def update_plots(self, iter_train_loss_list, iter_train_acc_list, iter_val_loss_list, iter_val_acc_list):
        """
        This function is used to update the online training metric plots.
        Args:
            iter_train_loss_list: A list containing tuples of global iter and train loss on the iter.
            iter_train_acc_list: A list containing tuples of global iter and train accuracy on the iter.
            iter_val_loss_list: A list containing tuples of global iter and validation loss on the iter.
            iter_val_acc_list: A list containing tuples of global iter and validation accuracy on the iter.

        Returns: None. but redraws plot in notebook.
        """

        '''
        The loss metric plot.
        '''
        # Convert list to numpy array for easier indexing.
        iter_train_loss_list = np.asanyarray(iter_train_loss_list)

        # Plot the train loss.
        self.metric_plots['plt1'].plot(self.smooth(iter_train_loss_list[:, 0], smoothing_steps=10),
                                       self.smooth(iter_train_loss_list[:, 1], smoothing_steps=10),
                                       color="orange", label="train")

        # In case there is a validation list given.
        if len(iter_val_loss_list) > 0:
            # Convert list to numpy array for easier indexing.
            iter_val_loss_list = np.asanyarray(iter_val_loss_list)
            # Plot the validation loss.
            self.metric_plots['plt1'].plot(self.smooth(iter_val_loss_list[:, 0], smoothing_steps=10),
                                           self.smooth(iter_val_loss_list[:, 1], smoothing_steps=10),
                                           color="green", label="validation")

        # Set scale and labels.
        self.metric_plots['plt1'].set_yscale("log")
        self.metric_plots['plt1'].set_ylabel("Loss")
        self.metric_plots['plt1'].set_xlabel("Iters")

        # In case the legend is not yet initialized do it.
        if not self.metric_plots['plt1_legend']:
            self.metric_plots['plt1_legend'] = self.metric_plots['plt1'].legend()

        '''
        The accuracy metric plot.
        '''

        # Convert list to numpy array for easier indexing.
        iter_train_acc_list = np.asanyarray(iter_train_acc_list)
        # Plot the train accuracy.
        self.metric_plots['plt2'].plot(iter_train_acc_list[:, 0],
                                       iter_train_acc_list[:, 1] * 100,
                                       color="orange", label="train"
                                       )

        # In case there is a validation list given.
        if len(iter_val_acc_list) > 0:
            # Convert list to numpy array for easier indexing.
            iter_val_acc_list = np.asanyarray(iter_val_acc_list)
            # Plot the validation accuracy.
            self.metric_plots['plt2'].plot(iter_val_acc_list[:, 0],
                                           iter_val_acc_list[:, 1] * 100,
                                           color="green", label="validation")

        # Set labels.
        self.metric_plots['plt2'].set_ylabel("Accuracy [%]")
        self.metric_plots['plt2'].set_xlabel("Iters")

        # In case the legend is not yet initialized do it.
        if not self.metric_plots['plt2_legend']:
            self.metric_plots['plt2_legend'] = self.metric_plots['plt2'].legend()

        '''
        Redraw the canvas.
        '''
        self.metric_plots['fig'].canvas.draw()

    @staticmethod
    def calculate_accuracy(labels, predictions):
        """
        A method used to claculate accuracy on a batch given labels and predictions.
        Args:
            labels: Labels of the batch.
            predictions: Predictions on the patch in percentage.

        Returns: Accuracy on the batch.

        """

        # Get the batch size.
        batch_size = predictions.shape[0]

        # Prepare predictions and labels.
        predictions = torch.flatten(predictions).round().type(torch.long)
        labels = torch.flatten(labels).type(torch.long)

        # Calculate accuracy
        acc = torch.sum(predictions == labels).item()
        acc /= batch_size

        return acc

    @staticmethod
    def smooth(x, smoothing_steps):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        smoothed = (cumsum[smoothing_steps:] - cumsum[:-smoothing_steps]) / float(smoothing_steps)
        return smoothed
