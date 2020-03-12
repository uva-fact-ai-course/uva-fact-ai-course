""" This module implements the Dataset class to quickly import datasets.

"""

import torch
import matplotlib.pyplot as plt


class Dataset:

    def get_sample(self, train=True, binary=False, show_image=False):
        """ Get a single sample from the dataset.

        train
            get sample from training set or test set
        binary
            threshold sample to make it binary
        show_image
            plot and show image when retrieving sample
        """
        if train:
            loader = self.train_loader
        else:
            loader = self.test_loader
        for images, labels in loader:
            sample_image = images[0]
            sample_label = labels[0]
            if binary:
                sample_image[sample_image < 0.5] = 0
                sample_image[sample_image >= 0.5] = 1
            if show_image:
                visual_img = sample_image.numpy()[0]
                plt.imshow(visual_img, cmap='gray')
                plt.show()

            norm_img = (sample_image-torch.min(sample_image)) / \
                (torch.max(sample_image) - torch.min(sample_image))

            # Centers the mean around 0, as required by the CEM paper.
            sample_image = norm_img - 0.5

            return sample_image, sample_label

    def get_sample_by_class(self, train=True, class_label=1, show_image=False):
        """ Get a sample from the dataset with a certain class.

        train
            get sample from training set or test set
        class_label
            the label of the sample that should be retrieved
        show_image
            plot and show image when retrieving sample
        """
        if train:
            data_list = self.train_list
        else:
            data_list = self.test_list
        for image, label in data_list:
            sample_image = image[0]
            sample_label = int(label)
            if sample_label == class_label:
                if show_image:
                    visual_img = sample_image.numpy()[0]
                    plt.imshow(visual_img, cmap='gray')
                    plt.show()
                # Centers the mean around 0, as required by the CEM paper.
                image -= 0.5
                return image

    def get_batch(self, binary=False, train=True):
        """ Get a batch from the the dataset.

        binary
            threshold sample to make it binary
        train
            get sample from training set or test set
        """
        if train:
            loader = self.train_loader
        else:
            loader = self.test_loader
        for images, labels in loader:
            if binary:
                images[images < 0.5] = 0
                images[images >= 0.5] = 1
            # Centers the mean around 0, as required by the CEM paper.
            images -= 0.5
            return images, labels
