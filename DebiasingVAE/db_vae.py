import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import seaborn as sns
from scipy.stats import norm

# Setting a plot style.
sns.set(style="darkgrid")


class Encoder(nn.Module):
    """
    This is the encoder class of the VAE used. It defines it's structure and it's forward method.
    """

    def __init__(self, output_dim, latent_dim, device):
        """
        The init method of the VAE Encoder that will initialize the encoder keeping the conv_block and the classifier
        as in the standard CNN model.
        Args:
            output_dim: The output dim of the classifier.
            latent_dim: The latent dim used by the VAE.
            device: The device used for compute, cuda of cpu.
        """
        super().__init__()

        # Save arguments in internal variables.
        self.n_outputs = output_dim
        self.latent_dim = latent_dim
        self.device = device

        # This shape is the one at the end of the conv_block and is later used for deconvolution.
        self.pre_flatten_shape = (None, 72, 1, 1)

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

        # The flattened output-shape of the CNN.
        flatten_cnn_out_shape = np.prod(self.pre_flatten_shape[1:])

        # The dimensions of the hidden layer of the classifier.
        self.n_hidden_classifier = {
            'hidden_1': flatten_cnn_out_shape,
            'hidden_2': 1000,
        }

        # The dimensions of the hidden layer of the latent mean.
        self.n_hidden_z_mean = {
            'hidden_1': flatten_cnn_out_shape,
            'hidden_2': 1000,
        }

        # The dimensions of the hidden layer of the latent log sigma.
        self.n_hidden_z_logsigma = {
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
            nn.Linear(self.n_hidden_classifier['hidden_1'], self.n_hidden_classifier['hidden_2']),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden_classifier['hidden_2']),

            # Output header
            nn.Linear(self.n_hidden_classifier['hidden_2'], self.n_outputs),
            nn.Sigmoid(),
        )

        # The latent mean header.
        self.z_mean = nn.Sequential(
            # Hidden layer.
            nn.Linear(self.n_hidden_z_mean['hidden_1'], self.n_hidden_z_mean['hidden_2']),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden_z_mean['hidden_2']),

            # Output header
            nn.Linear(self.n_hidden_z_mean['hidden_2'], self.latent_dim),
        )

        # The latent log sigma header.
        self.z_logsigma = nn.Sequential(
            # Hidden layer.
            nn.Linear(self.n_hidden_z_logsigma['hidden_1'], self.n_hidden_z_logsigma['hidden_2']),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden_z_logsigma['hidden_2']),

            # Output header
            nn.Linear(self.n_hidden_z_logsigma['hidden_2'], self.latent_dim),
        )

    def forward(self, input_data):
        """
        Performs forward path of the encoder.
        Args:
            input_data: The input image batch.

        Returns: mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.

        """

        # Feed through shared con_block.
        conv_activation = self.conv_stack(input_data)

        # make sure the variable is properly set
        assertion_text = "The defined pre-flatten shape is incorrect," \
                         " got {} but expected {}.".format(conv_activation.shape, self.pre_flatten_shape)
        assert (conv_activation.shape[1:] == self.pre_flatten_shape[1:]), assertion_text

        # Flatten for feed through fully connected headers.
        conv_activation = conv_activation.flatten(start_dim=1)

        # Feed through classification, mean and stdv head.
        prob = self.classifier(conv_activation)
        mean = self.z_mean(conv_activation)
        log_std = self.z_logsigma(conv_activation)

        return prob, mean, log_std


class Decoder(nn.Module):
    """
    This is the decoder class of the VAE used. It defines it's structure and it's forward method.
    """

    def __init__(self, latent_dim, output_dim, pre_flatten_shape):
        """
        The init method of the VAE decoder that will initialize the encoder keeping the conv_block and the classifier
        as the inverse of the standard CNN model.
        Args:
            latent_dim: The latent dim used by the VAE.
            output_dim: The output dim of the reconstruction, here the dimension of the images passed to the encoder.
            pre_flatten_shape: The shape the shall be passed to the first deconvolution layer.
        """
        super().__init__()

        # Save arguments in internal variables.
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.pre_flatten_shape = pre_flatten_shape

        # A helper variable to print one time shape warnings.
        self.warning_helper = True

        # The number of filters used in respective deconvolution layers, close to inverse of encoder.
        self.n_filters = {
            'deconv_1': 72,
            'deconv_2': 48,
            'deconv_3': 24,
            'deconv_4': 12,
            'deconv_5': 3
        }
        # The used kernel size.
        self.kernel_size = {
            'deconv_1': 5,
            'deconv_2': 5,
            'deconv_3': 5,
            'deconv_4': 5,
        }

        # The used kernel size.
        self.stride = {
            'deconv_1': 2,
            'deconv_2': 2,
            'deconv_3': 2,
            'deconv_4': 2,
        }

        # The flattened input-shape to the DeCNN.
        flatten_decnn_in_shape = np.prod(pre_flatten_shape[1:])

        # The dimensions of the hidden layer of the inverted latent mean.
        self.n_hidden_inv_z_mean = {
            'hidden_1': 1000,
            'hidden_2': flatten_decnn_in_shape,
        }

        # The up-convolution block.
        self.upconv_stack = nn.Sequential(

            # Conv block 1.
            nn.ConvTranspose2d(self.n_filters['deconv_1'], self.n_filters['deconv_2'],
                               kernel_size=self.kernel_size['deconv_1'], stride=self.stride['deconv_1']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['deconv_2']),

            # Conv block 2.
            nn.ConvTranspose2d(self.n_filters['deconv_2'], self.n_filters['deconv_3'],
                               kernel_size=self.kernel_size['deconv_2'], stride=self.stride['deconv_2']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['deconv_3']),

            # Conv block 3.
            nn.ConvTranspose2d(self.n_filters['deconv_3'], self.n_filters['deconv_4'],
                               kernel_size=self.kernel_size['deconv_3'], stride=self.stride['deconv_3']),
            nn.ReLU(),
            nn.BatchNorm2d(self.n_filters['deconv_4']),

            # Conv block 4.
            nn.ConvTranspose2d(self.n_filters['deconv_4'], self.n_filters['deconv_5'],
                               kernel_size=self.kernel_size['deconv_4'], stride=self.stride['deconv_4']),
            nn.Sigmoid(),
        )

        # The latent mean header.
        self.invert_latent_mean = nn.Sequential(
            # Hidden layer.
            nn.Linear(self.latent_dim, self.n_hidden_inv_z_mean['hidden_1']),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden_inv_z_mean['hidden_1']),

            # Output header
            nn.Linear(self.n_hidden_inv_z_mean['hidden_1'], self.n_hidden_inv_z_mean['hidden_2']),
            nn.ReLU(),
        )

    def forward(self, input_data):
        """
        The forward path of the decoder. It inverts the mapping of the encoder.
        Args:
            input_data: The latent mean.

        Returns: The reconstruction mean of a batch of images.

        """

        # First we need to upscale (invert the downscaling of the encoder) the latent mean.
        inverted_mean = self.invert_latent_mean(input_data)

        # Set the input shape of the deconv_block basd on the output of the conv_block in the encoder.
        batch_size = (input_data.shape[0],)
        deconv_input_shape = batch_size + self.pre_flatten_shape[1:]

        # Then we can reshape it for for de-convolution.
        inverted_mean = inverted_mean.view(deconv_input_shape)

        # Feed through the up-convolution block to get an image reconstruction.
        reconstruction_mean = self.upconv_stack(inverted_mean)

        # Assure that the channels of input and output are equal.
        assert reconstruction_mean.shape[1] == self.output_dim[0], 'The reconstruction must have the same number of ' \
                                                                   'channels as the input to the encoder.'

        # In case we have not exactly hit the output dimension we need to interpolate so that the can be calculated.
        if reconstruction_mean.shape[2:] != self.output_dim[1:]:
            # Print a warning fo the user, cause that interpolating is not perfect.
            if self.warning_helper:
                self.print_red(
                    "WARNING: Reconstruction shape {} is not equal to expected output shape {}!"
                    " \nWill be interpolated.".format(reconstruction_mean.shape[1:], self.output_dim)
                )
                self.warning_helper = False

            # Interpolate output to expected size.
            reconstruction_mean = nn.functional.interpolate(reconstruction_mean,
                                                            size=self.output_dim[1:],
                                                            mode="bilinear",
                                                            align_corners=False
                                                            )

        return reconstruction_mean

    @staticmethod
    def print_red(skk):
        """
        A helper function to print text in red.
        Args:
            skk: The text to be printed.

        Returns: None, but prints the passed text in red.

        """
        print("\033[91m{}\033[00m".format(skk))


class DbVae(nn.Module):
    """
    This is the main Debiasing VAE model that contain the encoder and decoder. It's forward path, training and
    validation routine.
    """

    def __init__(self, latent_dim, img_dim, num_classes,
                 kl_weight, reconstruction_weight, classification_weight):
        """
        The init of the VAE, initializes the encoder, decoder and other instances needed for training and
        evaluation.
        Args:
            latent_dim: The latent dim of the VAE.
            img_dim: The image dimension passed (channels x h x w).
            num_classes: The number of classes to predict.
            reconstruction_weight: A weighting function for the reconstruction loss.
            kl_weight: A weighting function for the KL divergence.
            classification_weight: A weighting function for the classification loss.
        """
        super().__init__()

        # Save passed arguments to internal attributes.
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

        # A global iteration count that can be used for plotting or other timings in sub functions.
        self.global_iter = 0
        # A placeholder that helps to print samples every epoch.
        self.num_train_iters_per_epoch = None
        # Global epoch counter.
        self.epoch = None
        # Placeholder for plot title.
        self.plot_title = None

        # Choose computing device according to availability.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Init the loss module for training.
        self.criterion = torch.nn.BCELoss(reduction='mean').to(self.device)

        # Init the encoder.
        self.encoder = Encoder(output_dim=self.num_classes,
                               latent_dim=self.latent_dim,
                               device=self.device)

        # Init the decoder.
        self.decoder = Decoder(self.latent_dim,
                               output_dim=self.img_dim,
                               pre_flatten_shape=self.encoder.pre_flatten_shape)

        # A placeholder for metric plots.
        self.metric_plots = dict()

    def debiasing_loss_function(self, x, x_pred, y, y_prob, mu, logsigma, kl_weight,
                                reconstruction_weight, classification_weight):
        """
        This is the debiasing loss function proposed in the paper. Consisting of the standard vae loss combined
        with classification loss. The VAE loss masked so that only faces are learned in the latent structure.
        Args:
            x: input images
            x_pred: reconstructed images
            y: class labels
            y_prob: class predictions.
            mu: latent mean
            logsigma: latent log sigma
            reconstruction_weight: A weighting function for the reconstruction loss.
            kl_weight: A weighting function for the KL divergence.
            classification_weight: A weighting function for the classification loss.

        Returns: The masked weighted loss for training and it's components.
        """

        '''
        The VAE loss part, must be masked so that it only contains loss from the face samples.
        '''
        # Use the training data labels to mask non-face VAE losses to zero.
        face_mask = y.flatten()

        # KL divergence loss.
        kl_div = 0.5 * torch.sum(torch.exp(logsigma) + mu ** 2 - 1.0 - logsigma, dim=1)
        # Mask the KL divergence.
        assert kl_div.shape == face_mask.shape, 'Shapes of mask and loss must match.'
        masked_kl_div = face_mask * kl_div
        # Get mean on KL divergence.
        masked_kl_div = torch.mean(masked_kl_div)
        # Scale the loss by it's weight.
        masked_kl_div *= kl_weight

        # Reconstruction loss on faces.
        assert x.shape == x_pred.shape, 'Shapes of reconstruction must match input.'
        reconstruction_loss = nn.functional.pairwise_distance(x.flatten(start_dim=1),
                                                              x_pred.flatten(start_dim=1),
                                                              p=2.0, keepdim=True).flatten()

        # Mask the the reconstruction loss.
        assert reconstruction_loss.shape == face_mask.shape, 'Shapes of mask and loss must match.'
        masked_reconstruction_loss = face_mask * reconstruction_loss
        # Get mean on reconstruction loss..
        masked_reconstruction_loss = torch.mean(masked_reconstruction_loss)
        # Scale the loss by it's weight.
        masked_reconstruction_loss *= reconstruction_weight

        # The VAE is the sum of teh KL loss and the reconstruction loss.
        masked_vae_loss = masked_reconstruction_loss + masked_kl_div

        # From time to time we print a reconstruction sample.
        if self.global_iter % self.num_train_iters_per_epoch == 0:
            # Get an face example out of the batch.
            image_id = 0
            for image_id in range(x.shape[0]):
                if face_mask[image_id] == 1:
                    break

            # Title of the figure.
            title = self.plot_title + ' - Epoch {}: Reconstruction sample.'.format(self.epoch)
            # Create a figure.
            fig = plt.figure(figsize=(9, 4.5), num=title)
            fig.suptitle(title)

            # Plot ground truth.
            plt1 = fig.add_subplot(1, 2, 1)
            image = x.detach().cpu()[image_id].permute(1, 2, 0)
            image = np.rot90(image, axes=(1, 0))
            plt1.imshow(image)
            plt1.set_title('Ground Truth')
            plt1.axis('off')

            # Plot reconstruction and ground truth.
            plt2 = fig.add_subplot(1, 2, 2)
            image = x_pred.detach().cpu()[image_id].permute(1, 2, 0)
            image = np.rot90(image, axes=(1, 0))
            plt2.imshow(image)
            plt2.set_title('Reconstruction')
            plt2.axis('off')

            # Draw the figure.
            fig.canvas.draw()

        '''
        The classification loss on the face detecion task.
        '''

        # Make sure that the shapes of predictions and labels match.
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        assert y_prob.shape == y.shape, 'The shapes of predictions and lables must match'

        # Calculate the mean classification loss.
        classification_loss = self.criterion(y_prob, y)
        # Scale the loss by it's weight.
        classification_loss = classification_weight * classification_loss

        '''
        The final loss is the combination of both 
        '''
        # The total loss is the mean of the weighted classification loss and the masked weighted vae loss.
        total_loss = classification_loss + masked_vae_loss

        return total_loss, classification_loss.item(), masked_kl_div.item(), masked_reconstruction_loss.item()

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images (from bernoulli) and the means for
        these bernoullis (as these are used to plot the data manifold).
        Args:
            n_samples: The number of samples to be drawn.

        Returns: Sampled images and their means.
        """

        # We first generate some latent samples from a normal distribution.
        latent_samples = torch.randn(n_samples, self.latent_dim).to(device=self.device)

        # Set the model in eval mode.
        assert not self.training, 'Model has to be in eval mode!'

        with torch.no_grad():
            # Get the reconstructed image means.
            im_means = self.decoder(latent_samples)
            # Sample the images from bernoulli.
            sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means

    def sample_gaussian(self, mean, std):
        """
        A simple function that samples a Gaussian using the reparameterization trick.
        Args:
            mean: The mean of the samples.
            std: The standard deviation.

        Returns: A sample from a Gaussian with mean and std.

        """
        # First we sample the noise from the Normal.
        epsilon = torch.randn_like(std).to(device=self.device)

        # Then we project using the mean and std of the encoder outputs.
        sample = mean + epsilon * std

        return sample

    def forward(self, x, y):
        """
        Given input, perform an encoding and decoding step.
        Args:
            x: input images
            y: labels

        Returns: The weighted debiasing loss a list holding it's components and the classification probabilities.
        """

        # We first encode the samples to retrieve the latent representation and classification predictions.
        prob, mean, log_std = self.encoder(x)

        # Get the standard deviation from it's log.
        std = torch.exp(log_std)

        # Now we need to sample from the Gaussian using our re-parametrization trick and the predicted means and std.
        latent = self.sample_gaussian(mean=mean, std=std)

        # Now we can decode the latent space again and retrieve the reconstruction.
        reconstruction = self.decoder(latent)

        # Finally, we have all the components to calculate our loss.
        loss, class_loss, kl_div, reconstruction_loss = self.debiasing_loss_function(x, reconstruction,
                                                                                     y, prob, mean, log_std,
                                                                                     self.kl_weight,
                                                                                     self.reconstruction_weight,
                                                                                     self.classification_weight)

        return loss, [class_loss, kl_div, reconstruction_loss], prob

    def get_latent_mu(self, images, batch_size=1024):
        """
        Gets the latent mean for all images needed for re-weighting the sampling process.
        Args:
            images: A data-loader of the dataset.
            batch_size: The size of the batches to be processed.

        Returns: numpy array with the latent means for the respective images.

        """
        # Set model in evaluation mode.
        self.eval()

        # Define a loader over the images to be processed.
        loader = DataLoader(images, batch_size=batch_size, shuffle=False)

        # The number of images in the dataset.
        num_images = len(images)

        # Placeholder for the latent mean of all images.
        mu = np.zeros((num_images, self.latent_dim))

        # A helper variable for indexing.
        start_idx = 0
        # Now go through all the images.
        for image_batch in tqdm(loader):
            # Get the images and convert to tensor as needed for input.
            x, _ = image_batch
            batch = torch.as_tensor(x, dtype=torch.float32).to(self.device)
            batch = batch.permute(0, 3, 2, 1)

            # Get the latent means for the respective images.
            _, batch_mu, _ = self.encoder(batch)
            batch_mu = batch_mu.detach().cpu()

            # Calculate a helper for slicing
            end_idx = min(start_idx + batch_size, num_images + 1)

            # Save the latent means of the batch to the correct place in the placeholder.
            mu[start_idx:end_idx] = batch_mu

            # Update indexing helper.
            start_idx += batch_size

        # Set model back in training mode.
        self.train()

        return mu

    def get_training_sample_probabilities(self, images, epoch, bins=10, smoothing_fac=0.0, model_name=""):
        """
        Get the sample probabilities for debiasing.
        Args:
            epoch: The current epoch number.
            images: A data-loader of a dataset.
            bins: Number of bins to create the histogram of latent distributions
            smoothing_fac: Factor to smooth the densitity function.
            model_name: Name of the model for plot title

        Returns: numpy array with the sample  probability of each image.

        """

        # We first need to get all the latent means of the dataset.
        mu = self.get_latent_mu(images)

        # A placeholder for the sampling probabilities for the images.
        training_sample_p = np.zeros(mu.shape[0])

        # Consider the distribution for each latent variable.
        for i in range(self.latent_dim):
            # Extract the latent means of the respective latent dimension.
            latent_distribution = mu[:, i]

            # Generate a histogram of the latent distribution.
            hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

            # Set the boundaries of the bins to -/+ infinity.
            bin_edges[0] = -float("inf")
            bin_edges[-1] = float("inf")
            # Use the digitize function to see in which bin each sample falls.
            bin_idx = np.digitize(latent_distribution, bin_edges)

            # Convert the hist density to a probability of the respective bin.
            hist_bin_probability = hist_density / np.sum(hist_density)

            # Invert the density function to compute the sampling probability!
            sample_probabilities = 1.0 / (hist_bin_probability[bin_idx - 1] + smoothing_fac)
            # Add 1 to remove negative probabilities and use Log Sum Trick
            log_sample_probabilities = np.log(sample_probabilities + 1)

            # At the end.
            training_sample_p += log_sample_probabilities

        # Overall normalization.
        training_sample_p /= np.sum(training_sample_p)

        # Plot a hist of current sample probabilities and image samples of the bins.
        self.plot_sample_prob_hist(training_sample_p.copy(), images, bins, epoch, model_name)

        return training_sample_p

    @staticmethod
    def plot_sample_prob_hist(training_sample_p, images, bins, epoch, model_name):
        """
        This function prints a histogram over the sample proabilities and image samples of each bin.
        Args:
            training_sample_p: The list of sample probabilities over faces in images.
            images: The dataloader to the faces.
            bins: The number of bins used in the histogram.
            epoch: The current epoch number.
            model_name: Name of the model for plot title

        Returns: None but creates a figure.
        """

        def normalize(v):
            """
            Private Function to normalize an array (L1 normalization)
            
            Args:
                v: Numpy array
            
            Returns:
                The normalized array
            """
            norm_array = np.linalg.norm(v, ord=1)
            if norm_array == 0:
                norm_array = np.finfo(v.dtype).eps
            return v / norm_array

        # The title of the figure and the plot.
        title = '{} - Epoch {}: Sample Probabilities and examples of bins.'.format(model_name, epoch)

        # Create a new figure and set title.
        fig = plt.figure(figsize=(9.5, 8), num=title)
        fig.suptitle(title)

        # Add the histogram subplot checking
        if images.__class__.__name__ == "DummyMIT":
            # MIT data make the subplot larger
            plt1 = plt.subplot2grid(shape=(12, 10), loc=(0, 0), colspan=10, rowspan=5, fig=fig)
        else:
            # OURDataset data make the col half to allocate new plot
            plt1 = plt.subplot2grid(shape=(12, 11), loc=(0, 0), colspan=5, rowspan=5, fig=fig)

        # Generate and plot a overall histogram of the latent distribution.
        num_samples_bins, bin_edges, _ = plt1.hist(training_sample_p, bins=10)

        # Set the x limits of the plot and labels + some other formatting.
        plt1.set_xlim(left=bin_edges[0], right=bin_edges[-1])
        plt1.set_ylabel("# samples")
        plt1.set_xlabel("Sample Probability")
        plt1.grid(True)
        plt1.set_yscale("log")
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # Write the absolute number of samples above bars.
        for i in range(bins):
            plt.text(bin_edges[i], num_samples_bins[i], str(int(num_samples_bins[i])), fontsize=8)

        # Set the boundaries of the bins to -/+ infinity for itemizing.
        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")

        # If OurDataset, plot the race distribution over bins
        if images.__class__.__name__ != "DummyMIT":
            # Create a subplot to allocate the race distribution
            plt2 = plt.subplot2grid(shape=(12, 11), loc=(0, 6), colspan=5, rowspan=5, fig=fig)
            # See in which bin each sample falls.
            bin_indices = np.digitize(training_sample_p, bin_edges)
            # Select a palette for the plots
            palette = sns.color_palette("hls", 7)
            # Assign a color to each race
            race_color = {"Indian": palette[0], "Latino_Hispanic": palette[1], "White": palette[2], "Black": palette[3],
                          "Southeast Asian": palette[4], "East Asian": palette[5], "Middle Eastern": palette[6]}
            # Generate patches for the legend
            legend_patches = [mpatches.Patch(color=race_color[race], label=race) for race in race_color.keys()]
            # List of axes for futher processing if needed
            axes = list()

            # For each bin get all samples
            for bin_idx in range(1, bins + 1):
                # x axis for the bar plot
                r = np.arange(1)
                # Get all samples ID's from the bin
                samples_in_bin = np.where(bin_indices == bin_idx)[0]
                # Create a copy of the dataframe in the dataset
                df = images.data.copy()
                # Slice it so it only contains the samples in bin
                df = df.iloc[samples_in_bin]
                # Get the value counts by race
                df_plot = df.race.value_counts().sort_values(ascending=True)
                # Convert the index to a list of races
                race_names = df_plot.index.to_list()
                # Convert the values to a numpy array
                # Initialize the acumulative size for stack bars
                df_plot = normalize(df_plot.to_numpy(dtype='float64')) * 100.0
                cum_size = np.zeros(1)

                # For each race, stack bars
                for i, race in enumerate(race_names):
                    axes.append(plt2.bar(r + bin_idx - 1, df_plot[i], bottom=cum_size, color=race_color[race]))
                    cum_size += df_plot[i]

            # Get the plot position and create a box
            box = plt2.get_position()
            # Change plot position
            plt2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            # Put a legend below current axis
            plt2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
                        ncol=2, handles=legend_patches, fontsize=8)
            # Remove axes label
            plt2.xaxis.set_major_formatter(plt.NullFormatter())
            # Add y label.
            plt2.set_ylabel('Race Distribution in bin')

        # Use the digitize function to see in which bin each sample falls.
        bin_indieces = np.digitize(training_sample_p, bin_edges)

        # Now we can get four samples out of each bin.
        random_samples_indx_in_bins = []
        for bin_idx in range(1, bins + 1):
            # Get all samples in the bin.
            samples_in_bin = np.where(bin_indieces == bin_idx)[0]
            # Choose four random samples out of each bin if possible.
            if len(samples_in_bin) > 0:
                random_samples_in_bin = np.random.choice(samples_in_bin, size=5)
            else:
                random_samples_in_bin = []
            # Append to overall samples.
            random_samples_indx_in_bins.append(random_samples_in_bin)

        # Load the sampled images in memory.
        random_samples_in_bins = []
        # For the individual bins.
        for sample_indices_in_bin in random_samples_indx_in_bins:
            # A list holding the bin samples.
            random_samples_in_bin = []

            # For samples in the bin.
            for sample_indx in sample_indices_in_bin:
                # Get the image and the label.
                image, label = images[sample_indx]
                # Make sure it is a face.
                assert label == 1, "All samples must be faces."
                # Append it to the samples list.
                random_samples_in_bin.append(image)

            # Append the bin samples to the overall samples.
            random_samples_in_bins.append(random_samples_in_bin)

        # For the bins in the dataset.
        for col, images in enumerate(random_samples_in_bins):
            # List of subplots.
            subplot_lst = list()
            # Plot the individual images.
            for row, im in enumerate(images):
                # Create a subplot.
                subplot = plt.subplot2grid(shape=(12, 10), loc=(7 + row, col), colspan=1, rowspan=1, fig=fig)
                # Show the image.
                subplot.imshow(im)
                # Switch off the axis.
                subplot.axis("off")
                # append to list (Remove warning)
                subplot_lst.append(subplot)
            # Use the last appended figure to write the bin number belonging to the column
            if subplot_lst:  # skip if last appended figure was empty bin
                subplot_lst[-1].text(0.5, -0.3, "Bin {}".format(col + 1), size=12, ha="center",
                                     transform=subplot.transAxes)

        # Adjust the spacing and draw the canvas.
        fig.subplots_adjust(wspace=0, hspace=0, bottom=0.05, top=0.95)
        fig.canvas.draw()

    def sample_latent_space(self, n_samples=25, epoch=0):
        """
        This function samples the VAE latent space.
        Args:
            n_samples: The number of manifolds to be sampled.
            epoch: The epoch the model is in.

        Returns: None, but saves samples to disk.
        """

        # Number of rows in the plot.
        nrows = 5

        # Set the model in eval mode.
        assert not self.training, 'Model has to be in eval mode!'

        with torch.no_grad():
            # Get the latent samples.
            samples, _ = self.sample(n_samples)

            # Create a grid for printing.
            samples = make_grid(samples.view(n_samples, 3, 64, 64), nrow=nrows)

            # Prepare for saving.
            samples = samples.cpu().numpy().transpose(2, 1, 0)

            # The title of the figure and the plot.
            title = self.plot_title + ' Epoch {}: Latent Space samples.'.format(epoch)

            # Create a new figure and set title.
            fig = plt.figure(figsize=(9.5, 4), num=title)
            fig.suptitle(title)

            # Print the samples.
            plt.imshow(samples)
            plt.grid(False)
            plt.axis("off")

            # Adjust the spacing and draw the canvas.
            fig.subplots_adjust(wspace=0, hspace=0, bottom=0, top=0.90)
            fig.canvas.draw()

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

    def run_training(self, batch_size, num_epochs, train_dataset,
                     enable_debiasing=True, smoothing_fac=0.0,
                     title='VDB-VAE Training', validation_dataset=None):
        """
        The training function of the CNN. If a validation dataset is passed the validation epoch is evenly
        split across the training epoch. So that both epoch end at the same time.
        Args:
            batch_size: Batch size to be used in training and validation.
            num_epochs: Number of epochs to train.
            train_dataset: The dataset loader for trainingset.
            title: The title of the training stats figure.
            validation_dataset: If given the loader of the validation dataset.
            enable_debiasing: Recompute data sampling probabilities if True
            smoothing_fac: Factor to smooth the densitity function.

        Returns: A dict holding, the training and validation loss and accuracy lists over iterations
        """

        self.plot_title = title

        # Set the model in training mode.
        self.train()

        # The learning rate to be used in the optimizer.
        learning_rate = 1e-4

        # Set up the optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.01)

        # Init of the metric plots that show the training progress.
        self.metric_plots['fig'] = plt.figure(figsize=(9, 10.5), num=title)
        self.metric_plots['plt1'] = self.metric_plots['fig'].add_subplot(3, 1, 1)
        self.metric_plots['plt1_legend'] = None
        self.metric_plots['plt2'] = self.metric_plots['fig'].add_subplot(3, 1, 3)
        self.metric_plots['plt2_legend'] = None
        self.metric_plots['plt3'] = self.metric_plots['fig'].add_subplot(3, 1, 2)
        self.metric_plots['plt3_legend'] = None
        self.metric_plots['fig'].suptitle(title, y=0.98)
        self.metric_plots['fig'].subplots_adjust(bottom=0.05, top=0.95)

        # Init a train loader that shuffles the data and returns batches of size batch_size.
        # The last batch is dropped so that we have no problems with the batch norm.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # We calculate the numbers of iterations in each epochs.
        self.num_train_iters_per_epoch = len(train_dataset) // batch_size
        # Then we can get the overall number of iterations in the training process, which is used to measure progress.
        num_train_iters = self.num_train_iters_per_epoch * num_epochs

        # In case we passed a validation dataset we also need to prepare it's usage.
        if validation_dataset:
            # Define a validation data loader similar as for training above.
            val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            # Calculate the iterations per epoch of validation.
            num_val_iters_per_epoch = len(validation_dataset) // batch_size

            # Then we space the evaluation iterations evenly across a training epoch, so that they finish together.
            val_iters_2_train_iters = np.linspace(start=0, stop=self.num_train_iters_per_epoch,
                                                  num=num_val_iters_per_epoch,
                                                  endpoint=False,
                                                  dtype=int)

        # In case there is no validation dataset passed, we just define no validation iters and a dummy val loader.
        else:
            val_iters_2_train_iters = []
            val_loader = None

        # Theses lists hold the loss and accuracy metrics over training iters.
        iter_val_loss_list = list()
        iter_val_component_loss_list = list()
        iter_val_acc_list = list()

        iter_train_loss_list = list()
        iter_train_component_loss_list = list()
        iter_train_acc_list = list()

        # We initialize a progressbar over total number of iterations for training.
        pbar = tqdm(total=num_train_iters, position=0, unit="it", leave=True)

        # Get All_Faces and All_Objects Datasets for using with debiasing
        all_faces = train_dataset.get_all_train_faces()
        all_objects = train_dataset.get_all_train_objects()

        # The main training loop starts here:
        for epoch in range(num_epochs):
            # Make epoch available globally.
            self.epoch = epoch

            # Set training mode.
            self.train()

            # In case we use debiasing and not first epoch.
            if enable_debiasing:
                # Make sure we can split batch in half.
                assert batch_size % 2 == 0, "The batch size must be a factor of two so it can be split in half."

                print("Recomputing debiasing sample probabilities..")

                # Recompute data sampling probabilities of faces in the dataset.
                p_faces = self.get_training_sample_probabilities(all_faces, epoch=epoch, smoothing_fac=smoothing_fac,
                                                                 model_name=title)

                # Init a sampler for faces, based on those probabilities.
                face_sampler = torch.utils.data.sampler.WeightedRandomSampler(p_faces, len(p_faces))
                # Init a dataloader for faces that returns half the batch given the sampler.
                train_loader_faces = DataLoader(all_faces, batch_size=batch_size // 2,
                                                sampler=face_sampler, drop_last=True)

                # Init a uniform dataloader for non-face objects that fills the second half of the batch.
                train_loader_objects = DataLoader(all_objects, batch_size=batch_size // 2,
                                                  shuffle=True, drop_last=True)

                # Zip the two loaders for iteration.
                train_loader = zip(train_loader_faces, train_loader_objects)

            # Sample from the latent space.
            self.eval()
            self.sample_latent_space(25, epoch)
            self.train()

            # In case of an validation dataset we initialize a iteration function over it's content.
            if validation_dataset:
                val_loader_iter = iter(val_loader)

            # For all batches in the train loader.
            for iter_id, sample_batched in enumerate(train_loader):

                # Unpack batch of training data based on debiasing active.
                if not enable_debiasing:
                    # In the case no debiasing case batches are already mixed.
                    image_batch, image_labels = sample_batched

                else:
                    # Extract face and non face batches
                    face_sample, object_sample = sample_batched

                    # Extract images and labels.
                    face_images, face_labels = face_sample
                    object_images, object_labels = object_sample

                    # Concatenate images and labels.
                    image_batch = torch.cat((face_images, object_images), 0)
                    image_labels = torch.cat((face_labels, object_labels), 0)

                    # Randomly shuffle the batch.
                    shuffle = torch.randperm(batch_size)
                    image_batch = image_batch[shuffle]
                    image_labels = image_labels[shuffle]

                # Convert the input images and labels to tensors.
                image_batch = torch.as_tensor(image_batch, dtype=torch.float32).to(self.device)
                image_labels = torch.as_tensor(image_labels, dtype=torch.float32).to(self.device)
                # Permute the image channels so that they are as expected in pytorch (channels x h x w).
                image_batch = image_batch.permute(0, 3, 2, 1)

                # Assure that the first dimension of the batch equals the set batch_size.
                assert image_batch.shape[0] == batch_size

                # Call the model with the image data to get the prediction probabilities and losses.
                loss, loss_components, face_prob = self.forward(image_batch, image_labels)

                # Reset the gradients in the network.
                optimizer.zero_grad()
                # Backprob the debiasing loss through the network.
                loss.backward()
                # Make a step of optimization.
                optimizer.step()

                # Calculate the accuracy in this iteration.
                iter_train_acc = self.calculate_accuracy(labels=image_labels, predictions=face_prob)

                # Calculate the global iteration counter used for plotting.
                self.global_iter = epoch * self.num_train_iters_per_epoch + iter_id

                # Save the training acc and loss to the respective lists.
                iter_train_loss_list.append([self.global_iter, loss.item()])
                iter_train_component_loss_list.append([self.global_iter, loss_components])
                iter_train_acc_list.append([self.global_iter, iter_train_acc])

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
                    image_labels = torch.as_tensor(image_labels, dtype=torch.float32).to(self.device).flatten()
                    # Permute the image channels so that they are as expected in pytorch (channels x h x w).
                    image_batch = image_batch.permute(0, 3, 2, 1)

                    # Without using gradient related computations.
                    with torch.no_grad():
                        # Assert batch size is as expected.
                        assert image_batch.shape[0] == batch_size

                        # Call the model with the image data to get the prediction probabilities and losses.
                        loss, loss_components, face_prob = self.forward(image_batch, image_labels)

                        # Calculate the accuracy in this iteration.
                        iter_val_acc = self.calculate_accuracy(labels=image_labels, predictions=face_prob)

                    # Save the validation acc and loss to the respective lists.
                    iter_val_loss_list.append([self.global_iter, loss.item()])
                    iter_val_component_loss_list.append([self.global_iter, loss_components])
                    iter_val_acc_list.append([self.global_iter, iter_val_acc])

                    # Set the model in training mode again.
                    self.train()

                # Every 200 iterations we redraw the training metric plots.
                if iter_id % 200 == 0:
                    self.update_plots(
                        iter_train_loss_list=iter_train_loss_list,
                        iter_train_components_loss_list=iter_train_component_loss_list,
                        iter_train_acc_list=iter_train_acc_list,
                        iter_val_loss_list=iter_val_loss_list,
                        iter_val_components_loss_list=iter_val_component_loss_list,
                        iter_val_acc_list=iter_val_acc_list,
                    )

                # We update the overall training progress of the model.
                pbar.update()

            if self.latent_dim == 2:
                self.generate_manifold(self.epoch)
            # Calculate the mean training loss and acc on the epoch for print statements.
            mean_train_acc_epoch = np.asanyarray(iter_train_acc_list[-self.num_train_iters_per_epoch:])[:, 1].mean()
            mean_train_loss_epoch = np.asanyarray(iter_train_loss_list[-self.num_train_iters_per_epoch:])[:, 1].mean()
            # Calculate the mean classification, KL and reconstruction loss.
            iter_train_component_loss = [item[1] for item in iter_train_component_loss_list]
            iter_train_component_loss = np.asanyarray(iter_train_component_loss)[-self.num_train_iters_per_epoch:]
            mean_train_class_loss_epoch = iter_train_component_loss[:, 0].mean()
            mean_train_kl_loss_epoch = iter_train_component_loss[:, 1].mean()
            mean_train_reconstruction_loss_epoch = iter_train_component_loss[:, 2].mean()

            # In case of validation.
            if validation_dataset:
                # Calculate the mean validation loss and acc on the epoch for print statements.
                mean_val_acc_epoch = np.asanyarray(iter_val_acc_list[-num_val_iters_per_epoch:])[:, 1].mean()
                mean_val_loss_epoch = np.asanyarray(iter_val_loss_list[-num_val_iters_per_epoch:])[:, 1].mean()
                # Calculate the mean classification, KL and reconstruction loss.
                iter_val_component_loss = [item[1] for item in iter_val_component_loss_list]
                iter_val_component_loss = np.asanyarray(iter_val_component_loss)[-num_val_iters_per_epoch:]
                mean_val_class_loss_epoch = iter_val_component_loss[:, 0].mean()
                mean_val_kl_loss_epoch = iter_val_component_loss[:, 1].mean()
                mean_val_reconstruction_loss_epoch = iter_val_component_loss[:, 2].mean()

                # Print epoch stats including validation.
                print(
                    "\n\n Epoch {:0>2d}/{:0>2d}:\n"
                    "   mean accuracy   train/val = {:.2%}   / {:.2%} \n"
                    "   mean train loss train/val = {:.2e} / {:.2e} \n"
                    "   mean class loss train/val = {:.2e} / {:.2e} \n"
                    "   mean KL loss    train/val = {:.2e} / {:.2e} \n"
                    "   mean recon loss train/val = {:.2e} / {:.2e} ".format(
                        epoch + 1,
                        num_epochs,
                        mean_train_acc_epoch,
                        mean_val_acc_epoch,
                        mean_train_loss_epoch,
                        mean_val_loss_epoch,
                        mean_train_class_loss_epoch,
                        mean_val_class_loss_epoch,
                        mean_train_kl_loss_epoch,
                        mean_val_kl_loss_epoch,
                        mean_train_reconstruction_loss_epoch,
                        mean_val_reconstruction_loss_epoch
                    )
                )

            else:
                # Print epoch stats.
                print(
                    "\n\n Epoch {:0>2d}/{:0>2d}:\n"
                    "   mean accuracy   train/val = {:.2%}   / -- \n"
                    "   mean train loss train/val = {:.2e} / -- \n"
                    "   mean class loss train/val = {:.2e} / -- \n"
                    "   mean KL loss    train/val = {:.2e} / -- \n"
                    "   mean recon loss train/val = {:.2e} / -- ".format(
                        epoch + 1,
                        num_epochs,
                        mean_train_acc_epoch,
                        mean_train_loss_epoch,
                        mean_train_class_loss_epoch,
                        mean_train_kl_loss_epoch,
                        mean_train_reconstruction_loss_epoch
                    )
                )

        # Close the progress bar at the end of the training
        pbar.close()

        # If debiasing get the probabilities once more to see the final dist.
        if enable_debiasing:
            print("Computing final debiasing sample probabilities..")
            # Recompute data sampling probabilities of faces in the dataset.
            _ = self.get_training_sample_probabilities(all_faces, epoch='Final', smoothing_fac=smoothing_fac)
            if self.latent_dim == 2:
                self.generate_manifold("Final")

        # Create the training/validation stats dictionary.
        stats = {
            'iter_train_loss_list': iter_train_loss_list,
            'iter_train_components_loss_list': iter_train_component_loss_list,
            'iter_train_acc_list': iter_train_acc_list,
            'iter_val_loss_list': iter_val_loss_list,
            'iter_val_components_loss_list': iter_val_component_loss_list,
            'iter_val_acc_list': iter_val_acc_list
        }

        return stats

    def generate_manifold(self, epoch):
        """
        Generate and plot the learned  latent manifold. Only usable if latent dimensions equal 2.
        
        Args:
            epoch: The epoch the model is in.
        """
        assert self.latent_dim == 2, "The latent dimensions must be 2."
        self.eval()

        # First we need to create a grid.
        num_steps = 20
        step_size = 0.01
        x_space = np.linspace(step_size, 1 - step_size, num_steps)
        y_space = np.linspace(step_size, 1 - step_size, num_steps)

        # Then we can apply the inverse ppf.
        x_ppf = norm.ppf(x_space)
        y_ppf = norm.ppf(y_space)

        # We go over the points in space.
        im_means = torch.empty(size=(num_steps, num_steps, 3, 64, 64))
        for x_id, x in enumerate(x_ppf):
            for y_id, y in enumerate(y_ppf):
                latent_sample = torch.as_tensor(data=[x, y]).view(1, 2).to(device=self.device)
                im_mean = self.decoder(latent_sample)
                im_means[x_id, y_id] = im_mean[0]

        im_means = im_means.detach().clone().to(device='cpu')

        # We need to reshape the flatten images back to 2D
        im_means = im_means.view(num_steps ** 2, 3, 64, 64)

        # Now we can use the make_grid plot function.
        grid_plot_mean = make_grid(im_means, nrow=num_steps)
        # We have to change the dimensions so it's right for matplotlib.
        grid_plot_mean = np.rot90(grid_plot_mean.permute(1, 2, 0), axes=(1, 0))

        # And do some formatting plus saving.
        title = self.plot_title + ' Epoch {}: Latent manifold.'.format(epoch)

        # Create a new figure and set title.
        fig = plt.figure(figsize=(9.5, 4), num=title)
        fig.suptitle(title)

        # Print the samples.
        plt.imshow(grid_plot_mean)
        plt.grid(False)
        plt.axis("off")

        # Adjust the spacing and draw the canvas.
        fig.subplots_adjust(wspace=0, hspace=0, bottom=0, top=0.90)
        plt.show()
        fig.canvas.draw()

        self.train()

    def update_plots(self, iter_train_loss_list, iter_train_components_loss_list, iter_train_acc_list,
                     iter_val_loss_list, iter_val_components_loss_list, iter_val_acc_list):
        """
        This function is used to update the online training metric plots.
        Args:
            iter_train_loss_list: A list containing tuples of global iter and train loss on the iter.
            iter_train_components_loss_list: A list containing tuples of global iter
                                             and components of the vae training loss on the iter.
            iter_train_acc_list: A list containing tuples of global iter and train accuracy on the iter.
            iter_val_loss_list: A list containing tuples of global iter and validation loss on the iter.
            iter_val_components_loss_list: A list containing tuples of global iter
                                           and components of the vae validation loss on the iter.
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
        The component loss metric plot.
        '''

        # The plot indices.
        iter_indices = [item[0] for item in iter_train_components_loss_list]
        iter_indices = self.smooth(iter_indices, smoothing_steps=10)
        iter_component_loss = [item[1] for item in iter_train_components_loss_list]
        # Convert list to numpy array for easier indexing.
        iter_component_loss = np.asanyarray(iter_component_loss)

        # Extract the individual loss terms.
        iter_train_class_loss = self.smooth(iter_component_loss[:, 0], smoothing_steps=10)
        iter_train_kl_loss = self.smooth(iter_component_loss[:, 1], smoothing_steps=10)
        iter_train_reconstruction_loss = self.smooth(iter_component_loss[:, 2], smoothing_steps=10)

        # Plot the loss terms.
        palette = sns.color_palette("OrRd", 3)
        self.metric_plots['plt3'].plot(iter_indices, iter_train_class_loss,
                                       color=palette[0], label="train classification")
        self.metric_plots['plt3'].plot(iter_indices, iter_train_kl_loss,
                                       color=palette[1], label="train KL divergence")
        self.metric_plots['plt3'].plot(iter_indices, iter_train_reconstruction_loss,
                                       color=palette[2], label="train reconstruction")

        # In case there is a validation list given.
        if len(iter_val_components_loss_list) > 0:
            # The plot indices.
            iter_indices = [item[0] for item in iter_val_components_loss_list]
            iter_indices = self.smooth(iter_indices, smoothing_steps=10)
            iter_component_loss = [item[1] for item in iter_val_components_loss_list]
            # Convert list to numpy array for easier indexing.
            iter_component_loss = np.asanyarray(iter_component_loss)

            # Extract the individual loss terms.
            iter_val_class_loss = self.smooth(iter_component_loss[:, 0], smoothing_steps=10)
            iter_val_kl_loss = self.smooth(iter_component_loss[:, 1], smoothing_steps=10)
            iter_val_reconstruction_loss = self.smooth(iter_component_loss[:, 2], smoothing_steps=10)

            # Plot the loss terms.
            palette = sns.color_palette("YlGn", 3)
            self.metric_plots['plt3'].plot(iter_indices, iter_val_class_loss,
                                           color=palette[0], label="val classification")
            self.metric_plots['plt3'].plot(iter_indices, iter_val_kl_loss,
                                           color=palette[1], label="val KL divergence")
            self.metric_plots['plt3'].plot(iter_indices, iter_val_reconstruction_loss,
                                           color=palette[2], label="val reconstruction")

        # Set scale and labels.
        self.metric_plots['plt3'].set_yscale("log")
        self.metric_plots['plt3'].set_ylabel("Component Loss")
        self.metric_plots['plt3'].set_xlabel("Iters")

        # In case the legend is not yet initialized do it.
        if not self.metric_plots['plt3_legend']:
            self.metric_plots['plt3_legend'] = self.metric_plots['plt3'].legend(ncol=3, prop={'size': 9})

        '''
        The accuracy metric plot.
        '''

        # Convert list to numpy array for easier indexing.
        iter_train_acc_list = np.asanyarray(iter_train_acc_list)
        # Plot the train accuracy.
        self.metric_plots['plt2'].plot(
            iter_train_acc_list[:, 0], iter_train_acc_list[:, 1] * 100, color="orange", label="train"
        )

        # In case there is a validation list given.
        if len(iter_val_acc_list) > 0:
            # Convert list to numpy array for easier indexing.
            iter_val_acc_list = np.asanyarray(iter_val_acc_list)
            # Plot the validation accuracy.
            self.metric_plots['plt2'].plot(
                iter_val_acc_list[:, 0], iter_val_acc_list[:, 1] * 100, color="green", label="validation"
            )

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
    def smooth(x, smoothing_steps):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        smoothed = (cumsum[smoothing_steps:] - cumsum[:-smoothing_steps]) / float(smoothing_steps)
        return smoothed
