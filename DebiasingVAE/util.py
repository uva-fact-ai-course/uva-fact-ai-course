import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import requests
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import torch
import tarfile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from create_our_dataset.util import plot_sample_images_dataset
from mpl_toolkits.axes_grid1 import ImageGrid
from stats_util import compare_mit_our_race_dist
import itertools
import concurrent.futures

# Setting a plot style.
sns.set(style="darkgrid")


class MitTrainingDatasetLoader(Dataset):
    """
    The MIT training dataset class.
    """

    def __init__(self, data_path):
        """
        The init method of the function loads the dataset.
        Args:
            data_path: Path to the MIT dataset.
        """

        print("Opening {}".format(data_path))
        sys.stdout.flush()

        # Open the dataset file.
        self.cache = h5py.File(data_path, "r")

        print("    Loading data into memory...")
        sys.stdout.flush()
        # Load the images in memory.
        self.images = self.cache["images"][:]
        self.labels = self.cache["labels"][:]

        # Get the number of samples.
        n_train_samples = self.images.shape[0]
        # Permute the order in the dataset.
        self.train_inds = np.random.permutation(np.arange(n_train_samples))

        # Label the dataset.
        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]

        print("    Done")

    def __len__(self):
        """
        The len function.
        Returns: Returns the lenght of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        The get_item function of the dataset, that loads images and labels and returns them.
        Args:
            idx: The index of the samples to return.

        Returns: The images and labels of the passed index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx, :, :, ::-1] / 255
        label = self.labels[idx]

        return image, label

    def get_all_train_faces(self):
        """
        This function returns a DummyDataset containing only the faces.
        Returns: DummyDataset containing only faces.
        """
        return DummyMIT(self.images[self.pos_train_inds], self.labels[self.pos_train_inds])

    def get_all_train_objects(self):
        """
        This function returns a DummyDataset containing only the counterexamples.
        Returns: DummyDataset containing only counterexamples.
        """
        return DummyMIT(self.images[self.neg_train_inds], self.labels[self.neg_train_inds])


class DummyMIT(Dataset):
    """
    This is a dummy dataset class for the MIT dataset, used to create sub-datasets of the MIT dataset.
    """

    def __init__(self, images, labels):
        """
        The init method.
        Args:
            images: The images that should be contained by the dataset.
            labels: The labels of respective images.
        """
        # Make a copy of the passed images and labels.
        self.images = images.copy()
        self.labels = labels.copy()

    def __len__(self):
        """
        The len function.
        Returns: Returns the lenght of the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        The get_item function of the dataset, that loads images and labels and returns them.
        Args:
            idx: The index of the samples to return.

        Returns: The images and labels of the passed index.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx, :, :, ::-1] / 255
        label = self.labels[idx]

        return image, label


class PPBFaceEvaluator(Dataset):
    """ The PPB dataset used for bias evaluation."""

    def __init__(self, path_to_faces):
        """
        The init method.
        Args:
            path_to_faces: Path to the dataset.
        """

        # Extract the folder location of the datset.
        folder = path_to_faces.split("/")[:-1]
        folder = "/".join(folder)

        print("Opening {}".format(path_to_faces))
        sys.stdout.flush()

        # Open the tar file and extract input.
        tf = tarfile.open(path_to_faces)
        tf.extractall(path=folder)

        print("    Loading data into memory...")
        sys.stdout.flush()

        # The folder containing the PPB data.
        self.ppb_root = os.path.join(os.path.split(path_to_faces)[0], "PPB-2017")

        # The path to the annotation file.
        ppb_anno = os.path.join(self.ppb_root, "PPB-2017-metadata.csv")
        # Read the CSV
        self.data = pd.read_csv(ppb_anno)

        # Update the header of the CSV and minor pre-processing
        self.data.columns = ["id", "file", "gender", "numeric", "skin_color", "country"]
        self.data.gender = self.data["gender"].str.lower()
        self.data.skin_color = self.data["skin_color"].str.lower()
        self.headers = list(self.data.columns.values)

        # Update img folder
        self.ppb_root = os.path.join(self.ppb_root, "imgs")

        print("    Done")

    def __len__(self):
        """
        The len function.
        Returns: Returns the lenght of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        The get_item function of the dataset, that loads images and labels and returns them.
        Args:
            idx: The index of the samples to return.

        Returns: The images and labels of the passed index.
        """
        # Convert tensor to list.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image name.
        img_name = os.path.join(self.ppb_root, self.data.iloc[idx, 1])
        # Load the image and normalize.
        image = cv2.imread(img_name)[:, :, [2, 1, 0]] / 255
        # Create a sample dict.
        sample = dict()
        sample["image"] = image
        # Add labels.
        for i in range(2, len(self.headers)):
            sample[self.headers[i]] = self.data.iloc[idx, i]

        return sample

    def evaluate(self,
                 models_to_test,
                 gender,
                 skin_color,
                 output_idx,
                 from_logit=False,
                 patch_stride=0.2,
                 patch_depth=5,
                 use_threading=True):
        """
        The evaluation function of a sub class in the PPB dataset using the sliding window approach.
        Args:
            use_threading: If True uses a threaded version to get the squares.
            models_to_test: The models to test on PPB.
            gender: The gender sub-class to evaluate.
            skin_color: The skin_color subclass to evaluate.
            output_idx: Index of the model output holding the face class probabilities.
            from_logit: If the output of the models are logits.
            patch_stride: Patch Stride to be used when applying the sliding window approach.
            patch_depth: Patch Depth to be used when applying the sliding window approach.

        Returns: The accuracy on the respective sub class.
        """

        # Helper variable to count correct predictions.
        correct_predictions = [0.0] * len(models_to_test)

        # Filter the data according to gender and skin_color.
        filtered = self.data[(self.data.gender == gender) & (self.data.skin_color == skin_color)]
        # Get the number of samples in sub-group.
        num_faces = len(filtered)

        # Set all models in evaluation mode.
        for model_idx, model in enumerate(models_to_test):
            model.eval()

        # For all evaluation
        for face_idx in tqdm(range(num_faces)):

            # Load the image.
            img_name = os.path.join(self.ppb_root, filtered.iloc[face_idx, 1])
            image = cv2.imread(img_name)[:, :, [2, 1, 0]]

            # Get the shape and construct the image batches by applying sliding window approach.
            height, width, _ = image.shape
            if use_threading:
                batches = slide_square_threaded(image, patch_stride, width / 2, width, patch_depth)
            else:
                batches = slide_square(image, patch_stride, width / 2, width, patch_depth)

            # Convert to tensors and normalize.
            batches = np.asanyarray(batches)
            batches_tensor = torch.as_tensor(batches, dtype=torch.float32) / 255.0
            # Permute channels to fit model.
            batches_tensor = batches_tensor.permute(0, 3, 2, 1)

            # Without the computations related to gradients.
            with torch.no_grad():
                # For every model to test.
                for model_idx, model in enumerate(models_to_test):

                    # Send batches tensor to device.
                    batches_tensor = batches_tensor.to(model.device)

                    # Get model output.
                    out = model.forward(batches_tensor)

                    # Extract class probabilites.
                    y = out if output_idx is None else out[output_idx]
                    y = y.cpu().detach().numpy()

                    # Get value on most likely face prediction.
                    y_inds = np.argsort(y.flatten())
                    max_ind = y_inds[-1]
                    y_max = y[max_ind]

                    # Check if prediction was face.
                    if (from_logit and y_max >= 0.0) or (not from_logit and y_max >= 0.5):
                        correct_predictions[model_idx] += 1

        # Calculate the accuracy.
        accuracy = [correct_predictions[i] / num_faces for i, _ in enumerate(models_to_test)]

        return accuracy


class OURDatasetLoader(Dataset):
    """
    This is OurDataset dataset class. It is used to initialize the dataset, to read it, to evaluate it, and anything
    else needed to conduct training and evaluation.
    """

    def __init__(self, csv_file, root_dir, dtype=None):
        """
        The init function of the dataset, linking to the raw data.
        Args:
            csv_file: The csv file holding the dataset infos, like labels and image names.
            root_dir: The root dir of the dataset.
            dtype: The datatype of the fields to read in the csv.
        """

        print("Opening {}".format(csv_file.split("/")[-1]))
        print("    Loading necessary info into memory...")
        sys.stdout.flush()

        # Read in the csv file as pandas.
        self.data = pd.read_csv(csv_file, dtype=dtype)
        # Set labels based on class annotation.
        self.data["is_face"] = np.where(self.data["image_class"] == "Human Face", 1, 0)
        # Set the root dir.
        self.root_dir = root_dir

        print("    Done")

    def __len__(self):
        """
        The len function.
        Returns: Returns the lenght of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        The get_item function of the dataset, that loads images and labels and returns them.
        Args:
            idx: The index of the samples to return.

        Returns: The images and labels of the passed index.
        """

        # Convert tensor to list.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image name.
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        # Read the image.
        image = cv2.imread(img_name)[:, :, [2, 1, 0]] / 255
        # Read the label.
        label = self.data.iloc[idx, -1]

        return image, label

    def create_biased_dataset(self, plot=True):
        """
        Method that generates a dataset simulating the MIT dataset distribution between race classes.
        Args:
            plot: True by default. Set false for no plotting the comparison between data

        Returns: A subset of OurDataset with an artificially generated bias.
        """
        print("    Generating artificially biased dataset...")

        '''
        Fill the dataset with faces based on the new biased distribution.
        '''

        # Get's all face data points.
        filtered = self.data[(self.data.is_face == 1)]
        # The percentage of samples to keep race classes.
        percentage = [0.03, 0.02, 0.07, 0.87, 0.02, 0.02, 0.01]
        # A placeholder for data frames.
        frames = list()

        # For each race in the dataset.
        for i, c in enumerate(filtered.race.unique()):
            # Filter the dataset by race.
            a = filtered[filtered.race == c]
            # Get the number of samples to keep in that class.
            value_number = np.floor(len(filtered) * percentage[i])

            # If we would need to take more than available, we take all samples
            if value_number >= len(a):
                a = a.copy()

            # Else we reduce the dataset, while keeping gender dist equal.
            else:
                # Take half of the size from female sub-class.
                female = a[a.gender == "Female"]
                female = female.iloc[: int(value_number / 2), :].copy()
                # Take half of the size from male sub-class.
                male = a[a.gender == "Male"]
                male = male.iloc[: int(value_number / 2), :].copy()
                # Combine them to one dataframe.
                a = pd.concat([male, female])

            # Save the data frame to the biased dataset.
            frames.append(a)

        '''
        Now we need to fill the dataset with counterexamples.
        '''

        # Get all the no face samples.
        no_face = self.data[(self.data.is_face == 0)]
        # The percentile to extract from each counter class so that we get nearly as much counter examples as faces.
        # Was found by quantitative.
        per_class_percentile = 0.4
        # For every counter class.
        for c in no_face.image_class.unique():
            # Get the samples.
            a = no_face[no_face.image_class == c]
            # Extract the percentile.
            a = a.iloc[: int(np.floor(len(a) * per_class_percentile)), :].copy()
            frames.append(a)

        '''
        Combine the two subsets.
        '''
        dataset = pd.concat(frames)
        print("    Done...")

        # Plot a race distribution comparison if needed.
        if plot:
            compare_mit_our_race_dist(dataset)

        return DummyDataset(dataset, self.root_dir)

    def get_all_train_faces(self):
        """
        This function returns a DummyDataset containing only the faces.
        Returns: DummyDataset containing only faces.
        """
        # Get all the faces.
        filtered = self.data[(self.data.is_face == 1)]

        return DummyDataset(filtered, self.root_dir)

    def get_all_train_objects(self):
        """
        This function returns a DummyDataset containing only the counterexamples.
        Returns: DummyDataset containing only counterexamples.
        """
        # Get all the counterexamples.
        filtered = self.data[(self.data.is_face == 0)]

        return DummyDataset(filtered, self.root_dir)

    def evaluate(self, model_to_test, age, gender, race, output_idx, verbose=False):
        """
        The evaluation function of OurDataset given a specific sub class.
        Args:
            model_to_test: The model to evaluate.
            age: The age class to evaluate.
            gender: The gender class to evaluate.
            race: The race class to evaluate.
            output_idx: Index of the model output holding the face class probabilities.
            verbose: If True print all infos.

        Returns: The accuracy on the respective sub class.

        """

        # The number of correct predictions.
        correct_predictions = 0.0

        # Get all the faces.
        only_faces = self.data[(self.data.is_face == 1)]

        '''
        Filter according to given sub classes.
        '''

        filtered = only_faces
        if verbose:
            print("Only Faces", len(filtered))

        # Filter by age if given.
        if age:
            filtered = filtered[(filtered.age == age)]
            if verbose:
                print("age filtered", len(filtered))

        # Filter by gender if given.
        if gender:
            filtered = filtered[(filtered.gender == gender)]
            if verbose:
                print("gender filtered", len(filtered))

        # Filter by race if given.
        if race:
            filtered = filtered[(filtered.race == race)]
            if verbose:
                print("race filtered", len(filtered))

        if verbose:
            print("Total filtered -> ", len(filtered))

        # Create a dummy dataset from the filtered faces.
        dummy_data = DummyDataset(filtered, self.root_dir)

        # The number of faces in that sub-dataset.
        num_faces = len(dummy_data)

        # Create a data loader for this sub-dataset.
        batch_loader = DataLoader(dummy_data, batch_size=16, shuffle=True)

        # Set model in eval mode.
        model_to_test.eval()

        # For all faces in the sub-dataset.
        for sample_batched in batch_loader:
            # Extract and convert batches.
            image_batch, _ = sample_batched
            image_batch = torch.as_tensor(image_batch, dtype=torch.float32)
            image_batch = image_batch.permute(0, 3, 2, 1)

            # Without gradient related computations.
            with torch.no_grad():
                # Get model output.
                image_batch = image_batch.to(model_to_test.device)
                out = model_to_test.forward(image_batch)

                # Extract class prediction.
                predictions = out if output_idx is None else out[output_idx]
                predictions = predictions.cpu().detach()

                # Check if correctly labeled as face.
                correct_predictions += torch.sum(torch.flatten(predictions).round().type(torch.long) == 1)

        # The accuracy on this sub-dataset.
        accuracy = correct_predictions / num_faces

        return accuracy


class DummyDataset(OURDatasetLoader):
    """
    This is a Dummy dataset class that can be used to create a dataset from a special subset of the main datasets,
    e.g. to make a only faces dataset.
    """

    def __init__(self, pandas_df, root_dir):
        """
        The init method of the dataset.
        Args:
            pandas_df: The pandas data frame holding the samples.
            root_dir: The root dir of the data.
        """
        # Store in internal attributes.
        self.data = pandas_df.copy()
        self.root_dir = root_dir

    def __len__(self):
        """
        The len function.
        Returns: Returns the lenght of the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        The get_item function of the dataset, that loads images and labels and returns them.
        Args:
            idx: The index of the samples to return.

        Returns: The images and labels of the passed index.
        """

        # Cast index tensor to list.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image name.
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        # Read the image and normalize it.
        image = cv2.imread(img_name)[:, :, [2, 1, 0]] / 255
        # Get the labels.
        label = self.data.iloc[idx, -1]

        return image, label


def slide_square(img, stride, min_size, max_size, n, target_im_shape=(64, 64, 3)):
    """
    Function to slide a number of squares across image and extract square regions. Interpolate squares to target size.
    Used to make images of different size fit the prediction model.
    Args:
        target_im_shape: the target image size of the squares to fit the model.
        img: the image
        stride: (0,1], provides the fraction of the dimension for which to slide to generate a crop
        min_size: minimum square size
        max_size: maximum square size
        n: number of different square sizes including min_size, max_size

    Returns: The squared and resized images.

    """

    # Get height and width of image.
    img_h, img_w = img.shape[:2]

    # Calculate the square sizes, in linear spaces.
    square_sizes = np.linspace(min_size, max_size, n, dtype=np.int32)

    # List holding the created squares.
    square_images = []

    # for each of the square_sizes
    for level, sq_dim in enumerate(square_sizes):

        # Calculate the stride length.
        stride_length = int(stride * sq_dim)

        # Get the pixel indices of respective strides.
        stride_start_i = range(0, int(img_h - sq_dim + 1), stride_length)
        stride_start_j = range(0, int(img_w - sq_dim + 1), stride_length)

        # For cut positions in height and width.
        for i in stride_start_i:
            for j in stride_start_j:
                # Crop square out of image.
                square_image = img[i: i + sq_dim, j: j + sq_dim]

                # Resize to target image size.
                square_resize = cv2.resize(square_image, target_im_shape[:2], interpolation=cv2.INTER_NEAREST)

                # append to list of images and bounding boxes
                square_images.append(square_resize)

    return square_images


def slide_square_threaded(img, stride, min_size, max_size, n, target_im_shape=(64, 64, 3)):
    """
    THREADED VERSION OF slide_square function.
    Function to slide a number of squares across image and extract square regions. Interpolate squares to target size.
    Used to make images of different size fit the prediction model.
    Args:
        target_im_shape: the target image size of the squares to fit the model.
        img: the image
        stride: (0,1], provides the fraction of the dimension for which to slide to generate a crop
        min_size: minimum square size
        max_size: maximum square size
        n: number of different square sizes including min_size, max_size

    Returns: The squared and resized images.

    """

    # Calculate the square sizes, in linear spaces.
    square_sizes = np.linspace(min_size, max_size, n, dtype=np.int32)

    # for each of the square_sizes start a thread.
    threads = list()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for level, sq_dim in enumerate(square_sizes):
            square_images_tread = executor.submit(slide_square_worker, img.copy(), stride, sq_dim, target_im_shape)
            threads.append(square_images_tread)

    # Wait until they terminate and get the old_results.
    square_images = list()
    for terminated_thread in concurrent.futures.as_completed(threads):
        square_images.append(terminated_thread.result())

    # Flatten the nested list to be batch-able.
    square_images = list(itertools.chain.from_iterable(square_images))

    return square_images


def slide_square_worker(img, stride, sq_dim, target_im_shape):
    """
    A slide square worker thread function.
    Args:
        img: The image to work on.
        stride: The stride to use.
        sq_dim: The square dimensiom.
        target_im_shape: The target image shape.

    Returns: The squared and resized images and their bounding boxes for the given square dim.

    """
    # Get height and width of image.
    img_h, img_w = img.shape[:2]

    # List holding the created squares.
    square_images = []

    # Calculate the stride length.
    stride_length = int(stride * sq_dim)

    # Get the pixel indices of respective strides.
    stride_start_i = range(0, int(img_h - sq_dim + 1), stride_length)
    stride_start_j = range(0, int(img_w - sq_dim + 1), stride_length)

    # For cut positions in height and width.
    for i in stride_start_i:
        for j in stride_start_j:
            # Crop square out of image.
            square_image = img[i: i + sq_dim, j: j + sq_dim]

            # Resize to target image size.
            square_resize = cv2.resize(square_image, target_im_shape[:2], interpolation=cv2.INTER_NEAREST)

            # append to list of images
            square_images.append(square_resize)

    return square_images


def download_url(url, filename=None):
    """
    Download a file from a url and place it in filename.
    Args:
        url: URL to download file from
        filename: Name to save the file under. If None, use the basename of the URL

    Returns: Path to file

    """

    # Extract the folder name
    folder = filename.split("/")[:-1]
    folder = "/".join(folder)

    # Create if not existing.
    if not os.path.exists(folder):
        os.mkdir(folder)

    # In case no filename is given set filename to base url.
    if not filename:
        filename = os.path.basename(url)

    # Check if file is already present locally, if not download.
    if not os.path.isfile(filename):
        print("    Not found. Starting download...")

        # Start a request session.
        with requests.get(url, stream=True) as r:
            # Get status of request.
            r.raise_for_status()

            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))

            # A progress bar for the download.
            progress = tqdm(total=total_size, unit="iB", unit_scale=True)

            # Open the file to write.
            with open(filename, "wb") as f:
                # For all chunks of the data.
                for chunk in r.iter_content(chunk_size=8192):
                    # filter out keep-alive new chunks
                    if chunk:
                        # Write chunk.
                        f.write(chunk)
                        # Update the progress bar.
                        progress.update(len(chunk))

            # Close the progress bar.
            progress.close()

        print("\n    Done")

    else:
        print("    Found offline.")

    return filename


def check_for_mit_dataset():
    """
    Checking for the MIT dataset.

    Returns: The pathes to MIT training and PPB validation dataset.
    """

    # Get the training data: both images from CelebA and ImageNet
    print("Checking for training dataset...")
    path_to_training_data = download_url(
        "https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1", "MIT_dataset/train_face.h5"
    )

    # Get the validation data: PPB images
    print("\nChecking for PPB test data_set...")
    path_to_test_data = download_url(
        "https://www.dropbox.com/s/l0lp6qxeplumouf/PPB.tar?dl=1", "MIT_dataset/test_face.tar"
    )

    print("\n All done!\n")

    return path_to_training_data, path_to_test_data


def check_for_our_dataset():
    """
    This function checks if our dataset is loaded and get's it if needed.

    Returns: The directory holding our data.
    """

    print("Checking for OUR dataset...")

    # The file id of our dataset.
    file_id = "1xWuZRX1ETO5BSr4Il0dXDSsQVMd-ndVg"

    # The destination of the file.
    destination = "our_dataset.tar.xz"

    # Download file if not present.
    download_file_from_google_drive(file_id, destination)

    # Print some samples of our data.
    get_sample_images_our_dataset("our_dataset/train")

    return "our_dataset/"


def get_sample_images_our_dataset(image_dir):
    """
    Get the samples on our dataset.
    Args:
        image_dir: The directory containing the images.

    Returns: None, but creates a plot.

    """

    # All the images in the directory.
    all_image_names = os.listdir(image_dir)

    # The faces and it's counterexamples.
    face_images = list()
    counter_examples = list()

    # We use the fact that dace images have much shorter names to sort all the images.
    for image_name in all_image_names:
        if len(image_name) < 10:
            face_images.append(image_name)
        else:
            counter_examples.append(image_name)

    # Plot the face examples.
    title = "Random training samples OUR Dataset of Class Face"
    plot_sample_images_dataset(image_dir, title, face_images)

    # Plot the counterexamples.
    title = "Random training samples OUR Dataset of Class NoFace"
    plot_sample_images_dataset(image_dir, title, counter_examples)


def show_examples_mit_train_data(dataset):
    """
    This function shows samples of the MIT training data.
    Args:
        dataset: The MIT training data.

    Returns: None, but plots the sample graphs.

    """
    # Define a dataloader to get some samples.
    dataloader = DataLoader(dataset, batch_size=300, shuffle=True)
    iterloader = iter(dataloader)

    # Get a batch of 300 samples.
    images, labels = next(iterloader)

    # Split samples in faces and counter examples.
    face_images = images[np.where(labels == 1)[0]]
    not_face_images = images[np.where(labels == 0)[0]]

    # Extract number of needed images.
    random_sample_face_images = face_images[-64:, :, :, :]
    random_sample_counter_images = not_face_images[-64:, :, :, :]

    # Helper lists to iterate over the needed plots.
    images_to_plot = [random_sample_face_images, random_sample_counter_images]
    plt_titles = [
        "Random training samples Mit Dataset of Class Face",
        "Random training samples Mit Dataset of Class NoFace",
    ]

    # For the two classes in the dataset.
    for plt_id, images in enumerate(images_to_plot):

        # Create the Figure and the image gird.
        fig = plt.figure(figsize=(9.0, 9.0), num=plt_titles[plt_id])
        grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.1)

        # Plot the individual images.
        for ax, im in zip(grid, images):
            ax.imshow(im)
            ax.axis("off")

        # Set the title and show the plot.
        plt.suptitle(plt_titles[plt_id], y=0.92, fontsize=18)
        # Draw the figure.
        fig.canvas.draw()
        plt.show()


def show_examples_mit_test_data():
    """
    This function plots a grid of samples out of PPB.

    Returns: None, but plots a figure.
    """
    # The image dir.
    image_dir = "MIT_dataset/PPB-2017/imgs"

    # List all images in dir.
    all_image_names = os.listdir(image_dir)

    # The figure title.
    title = "Random validation samples PPB validation set"

    # Sample and plot the images.
    plot_sample_images_dataset(image_dir, title, all_image_names)


def download_file_from_google_drive(file_id, destination):
    """
    This function downloads a file from Google Drive.
    Args:
        file_id: The file id of Google Drive.
        destination: The file destination.

    Returns: None, but writes the file to disk.

    """

    # A base URL used for the question.
    url = "https://docs.google.com/uc?export=download"

    # Check if file is already present locally.
    if not os.path.isdir("our_dataset"):
        print("    Not found. Starting download...")

        # Init a request session.
        session = requests.Session()

        # Get the Drive response.
        response = session.get(url, params={"id": file_id}, stream=True)
        # Get a confirm token for secure download.
        token = get_confirm_token(response)

        # In case we get a token, set up stream.
        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        # Start the actual downstream.
        save_response_content(response, destination)

        # Extract the tar file.
        print("    Extracting files...")
        tf = tarfile.open(destination)
        tf.extractall(path="")

        # Remove the original file.
        print("    Deleting tar file...")
        os.remove(destination)

        print("    All done!")

    else:
        print("    Found offline.")


def get_confirm_token(response):
    """
    A function used in the download process from Drive to get a confirm token.
    Args:
        response: The request response from the initial request.

    Returns: The confirm token

    """

    # Extract token.
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    """
    The actual download process for getting a file form Drive.
    Args:
        response: The response of the initial request.
        destination: The destination of the file to write.

    Returns: None, but writes file to disk.

    """

    # Set chunk size of download.
    chunk_size = 32768

    # This is the total size of our dataset.
    total_size = 188407104

    # A download progress bar.
    progress = tqdm(total=total_size, unit="iB", unit_scale=True)

    # We open a file.
    with open(destination, "wb") as f:
        # For all chunks.
        for chunk in response.iter_content(chunk_size):
            # Filter out keep-alive new chunks.
            if chunk:
                # Write chunk.
                f.write(chunk)
                # Update progress.
                progress.update(len(chunk))

    # Close the progress bar.
    progress.close()


def evaluate_on_ppb(model, test_loader, title, output_index=None, use_threading=True):
    """
    Evaluate the model on the PPB data.
    Args:
        use_threading: If True threading is used to apply sliding window approach.
        model: The model to evaluate.
        output_index: The index of the model return that holds the class probabilities.
        title: Title of plot and figure.
        test_loader: A data-loader of the PPB dataset.

    Returns: A dictionary of the respective sub accuracies and creates a bar plot of respective.

    """
    # A dictionary holding the stats for returning.
    accuracy_dict = dict()
    # A list holding the stats for plotting.
    accuracy_list = list()

    # For the in the date defined classes.
    for skin_color in ["lighter", "darker"]:
        for gender in ["male", "female"]:
            print("Evaluate standard model on {} skin {} subjects:".format(skin_color, gender))

            # Calculate the accuracy on respective subset.
            acc = test_loader.evaluate([model], gender, skin_color,
                                       from_logit=False, output_idx=output_index,
                                       use_threading=use_threading)[0]

            # Save to plot and return data structures.
            accuracy_dict[skin_color + '_' + gender] = acc
            accuracy_list.append(acc)

            print("    Accuracy {} {}: {:.2f}".format(skin_color, gender, acc))

    # Init a figure for the experiment old_results.
    fig = plt.figure(figsize=(9.5, 6), num=title)
    # Create a bar plot of the acc.
    plt.bar(range(4), accuracy_list)
    # Set the labels.
    plt.xticks(range(4), ("LM", "LF", "DM", "DF"))
    # Set y limits based on seen data.
    y_min = np.min(accuracy_list) - 0.1
    if y_min < 0.0:
        y_min = 0.0
    y_max = np.max(accuracy_list) + 0.03
    if y_max > 1.0:
        y_max = 1.01
    y_ticks = np.around(np.arange(start=y_min, stop=y_max, step=0.01), decimals=2)
    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks)
    # Set y label.
    plt.ylabel("Accuracy")
    # Set the title on the figure.
    plt.title(title)

    # Draw the figure.
    fig.canvas.draw()
    plt.show()

    return accuracy_dict


def evaluate_on_our_dataset(model, test_loader, do_race, do_gender, do_age, title, verbose=False, output_index=None):
    """
    This function evaluates the model on OurDataset. It creates all combinations on the defined boolean do flags.
    Args:
        model: The model to evaluate.
        output_index: Index of the model output holding the face class probabilities.
        verbose: If True print the individual sub classes.
        title: Title of figure and plot.
        test_loader: A data loader to our validation data.
        do_race: A boolean indicating if race label shall be used.
        do_gender: A boolean indicating if gender label shall be used.
        do_age: A boolean indicating if age label shall be used.

    Returns: A dictionary of the respective sub accuracies and creates a bar plot of respective.
    """

    # A dictionary holding the stats for returning.
    accuracy_dict = dict()
    # A list holding the stats for plotting.
    accuracy_list = list()

    # The race categories defined in our data.
    race = [
        "White",
        "Black",
        "Latino_Hispanic",
        "East Asian",
        "Indian",
        "Middle Eastern",
        "Southeast Asian",
    ]
    # The race ticks used for plotting.
    race_ticks = ["W", "B", "LH", "EA", "I", "ME", "SA"]

    # The gender categories defined in our data.
    gender = ["Male", "Female"]
    # The gender ticks used for plotting.
    gender_ticks = ["M", "F"]

    # The age categories defined in our data.
    age = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    # The age ticks used for plotting.
    age_ticks = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

    # A list holding all the individual used sub categories.
    pre_list = list()
    # Append the sub categories used.
    if do_race & do_gender & do_age:
        print("Do not put all 3 values to true. That is too much :)")
        return
    if do_race:
        pre_list.append(race)
    if do_gender:
        pre_list.append(gender)
    if do_age:
        pre_list.append(age)

    # A list of the final ticks.
    ticks = list()
    # Create a list of the combinations created out of individual sub categories.
    prod_list = list(itertools.product(*pre_list))

    # For each of the combinations defined.
    for combination_id, combination in enumerate(prod_list):
        # Init dummy variables
        r = None
        g = None
        a = None

        '''
        Unpacks the combinations accordingly and defines respective prints and ticks.
        '''
        if len(combination) == 2:
            # Unpack the two sub categories.
            t1, t2 = combination

            if do_race & do_gender:
                r = t1
                g = t2
                if verbose:
                    print("Evaluate standard model on {} race {} subjects:".format(r, g))
                text = "    Accuracy {} {}".format(r, g)
                ticks.append("".join([race_ticks[race.index(r)], gender_ticks[gender.index(g)]]))

            elif do_race & do_age:
                r = t1
                a = t2
                if verbose:
                    print("Evaluate standard model on {} race {} range subjects:".format(r, a))
                text = "    Accuracy {} {}".format(r, a)
                ticks.append("".join([race_ticks[race.index(r)], age_ticks[age.index(a)]]))

            else:
                g = t1
                a = t2
                if verbose:
                    print("Evaluate standard model on {} subjects with {} age:".format(g, a))
                text = "    Accuracy {} {}".format(g, a)
                ticks.append("".join([gender_ticks[gender.index(g)], age_ticks[age.index(a)]]))

        else:

            if do_race:
                r = combination[0]
                if verbose:
                    print("Evaluate standard model on {} race".format(r))
                text = "    Accuracy {}".format(r)
                ticks = race_ticks

            elif do_age:
                a = combination[0]
                if verbose:
                    print("Evaluate standard model on {} age subjects".format(a))
                text = "    Accuracy {}".format(a)
                ticks = age_ticks

            else:
                g = combination[0]
                if verbose:
                    print("Evaluate standard model on {}  subjects".format(g))
                text = "    Accuracy {}".format(g)
                ticks = gender_ticks

        '''
        Calculate the accuracy of respective combination or category.
        '''
        # Get accuracy.
        acc = test_loader.evaluate(model, a, g, r, output_idx=output_index)

        # Save to plot and return data structures.
        accuracy_list.append(acc)
        accuracy_dict[ticks[combination_id]] = acc.item()

        # Print the old_results of evaluation if verbose.
        if verbose:
            print(text + " {:.2%}\n".format(acc))

    # Create a figure for the old_results plot.
    fig = plt.figure(figsize=(9.5, 6), num=title)
    # Plot the bar plot of accuracies.
    plt.bar(range(len(prod_list)), accuracy_list)
    # Set the respective ticks
    plt.xticks(range(len(prod_list)), tuple(ticks))
    # Set the y tick based on the accuracies.
    y_min = np.min(accuracy_list) - 0.1
    if y_min < 0.0:
        y_min = 0.0
    y_max = np.max(accuracy_list) + 0.03
    if y_max > 1.0:
        y_max = 1.01
    y_ticks = np.around(np.arange(start=y_min, stop=y_max, step=0.01), decimals=2)
    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks)
    # Set the plot y label.
    plt.ylabel("Accuracy")
    # Set the title on the plot.
    plt.title(title)

    # Draw the figure.
    fig.canvas.draw()
    plt.show()

    return accuracy_dict


def get_bias_stats_on_our_dataset(model, test_loader, title, verbose=False, output_index=None):
    """
    This function call bias evaluations on OurDataset for race, gender, age.
    Args:
        model: The model to evaluate.
        output_index: Index of the model output holding the face class probabilities.
        verbose: If True print the individual sub classes.
        title: Title of figure and plot.
        test_loader: A data loader to our validation data.

    Returns: A dictionary of the  gender, race and age sub accuracies and creates a bar plot of respective.

    """
    # Init the return dict.
    accuracy_dict = dict()

    '''
    For Race
    '''
    race_stats = evaluate_on_our_dataset(model=model,
                                         test_loader=test_loader,
                                         do_race=True,
                                         do_gender=False,
                                         do_age=False,
                                         title=title + ': Race',
                                         verbose=verbose,
                                         output_index=output_index)
    accuracy_dict['race'] = race_stats

    '''
    For Gender
    '''
    gender_stats = evaluate_on_our_dataset(model=model,
                                           test_loader=test_loader,
                                           do_race=False,
                                           do_gender=True,
                                           do_age=False,
                                           title=title + ': Gender',
                                           verbose=verbose,
                                           output_index=output_index)
    accuracy_dict['gender'] = gender_stats

    '''
    For Age
    '''
    age_stats = evaluate_on_our_dataset(model=model,
                                        test_loader=test_loader,
                                        do_race=False,
                                        do_gender=False,
                                        do_age=True,
                                        title=title + ': Age',
                                        verbose=verbose,
                                        output_index=output_index)
    accuracy_dict['age'] = age_stats

    return accuracy_dict


def save_model(model, filename):
    """
    Save the given model as filename.pt

    Args:
        model: Pytorch model to be saved
        filename: Name of the model to be saved without extension
    """
    filename = "{}.pt".format(filename)
    torch.save(model.state_dict(), filename)
    print("Model save as {}".format(filename))


def set_seeds(seed):
    """
    This function sets all seeds in the used random functions.
    :param seed: The seed to set.
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
