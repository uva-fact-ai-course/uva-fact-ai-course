import os
from bidict import bidict
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm_notebook as tqdm
import csv
import zipfile
from distutils.dir_util import copy_tree
import shutil
import math
import copy


def check_raw_data(raw_fair_face_dir, raw_open_images_dir):
    """
    This function checks if the user provided raw data for dataset creation correctly.
    :param raw_fair_face_dir: The path to the raw FairFace images.
    :param raw_open_images_dir: The path to the raw OpenImages images.
    :return: None, but prints feedback.
    """

    # The data needed for FairFace.
    fair_face_docs_to_check = ['fairface-img-margin025-trainval.zip',
                               'fairface_label_train.csv',
                               'fairface_label_val.csv']

    # Get content of directory.
    dir_content = os.listdir(raw_fair_face_dir)
    print('Checking FairFace Files:')

    # Check if all specified files are there.
    for doc in fair_face_docs_to_check:
        if doc not in dir_content:
            raise ValueError("File {} is missing in {}".format(doc, raw_fair_face_dir))

        print("    Found {}".format(doc))

    print("    All good!\n")

    # The data needed for OpenImages.
    open_image_docs_to_check = ['test-annotations-bbox.csv',
                                'class-descriptions-boxable.csv',
                                'validation-annotations-bbox.csv',
                                'raw_open_images']

    # Get content of directory.
    dir_content = os.listdir(raw_open_images_dir)
    print('Checking OpenImages Files:')

    # Check if all specified files are there.
    for doc in open_image_docs_to_check:
        if doc not in dir_content:
            raise ValueError("File/Folder {} is missing in {}".format(doc, raw_fair_face_dir))

        print("    Found {}".format(doc))

    # Check if all the OpenImages have the correct extension. For sanity.
    image_dir_content = os.listdir(raw_open_images_dir + '/raw_open_images')
    if not all('.jpg' in filename for filename in image_dir_content):
        raise ValueError('Folder raw_open_images is not only containing jpg image files.')

    # Check if all the OpenImages have been provided based on the expected number. For sanity.
    number_images = len(image_dir_content)
    number_expected_images = 125436 + 41620  # Test and Validation size according to website
    if number_images != number_expected_images:
        raise ValueError('Expected {} images but found {}!'.format(number_expected_images, number_images))

    print("        Checked number of images")
    print("    All good!\n")


def remove_classes_from_open_images_dataset(raw_open_images_dir, classes_to_remove):
    """
    This function removes classes specified in classes_to_remove and images with no class
    from the original OpenImages dataset. This is done based on the bounding box labels of the data.
    Here this is used to make sure that no human faces are in the data anymore.
    :param raw_open_images_dir: The path to the raw OpenImages images.
    :param classes_to_remove: A list of class names to remove from data.
    :return: a list of the filtered content of the directory
    """

    # Get the content of the raw image dir.
    image_dir = raw_open_images_dir + '/raw_open_images'
    image_dir_content = os.listdir(image_dir)

    # Load the mapping from classname to classids.
    classname_2_classid = load_classname_2_classid_dict(raw_open_images_dir)

    # Load the mapping from image names to their bounding boxes.
    imageid_2_bboxes = load_imageid_2_bboxes_dict(raw_open_images_dir)

    # Convert the provided list of class_names to respective class ids.
    image_class_ids_to_remove = [classname_2_classid[classname] for classname in classes_to_remove]

    # Init a list holding valid images.
    filtered_image_dir_content = list()

    # Keep track of the removed images for demonstration purpose.
    counter_removed_no_bboxes = list()
    counter_removed_invalid_class = list()

    # Go through all the images.
    for image_name in image_dir_content:

        # Extract the image id from the name
        image_id = image_name.replace('.jpg', '')

        # Get the bounding boxes.
        bounding_boxes_image = imageid_2_bboxes[image_id]

        # If there are no labels on the image we discard it and go to the next sample.
        if len(bounding_boxes_image) == 0:
            counter_removed_no_bboxes.append(image_name)
            continue

        # If there are annotated boxes we get their labels.
        class_ids_in_image = [bbox[0] for bbox in bounding_boxes_image]

        # If there is any bounding box label that matches one of our classes_to_remove we discard the sample and
        # continue to the next.
        if any(image_class_id_to_remove in class_ids_in_image for image_class_id_to_remove in
               image_class_ids_to_remove):
            counter_removed_invalid_class.append(image_name)
            continue

        # If it is annotated and does not contain a forbidden class, we add it as valid.
        filtered_image_dir_content.append(image_name)

    print('Reduced number of images from {} to {} by deleting {} without bounding boxes'
          ' and {} of invalid classes.'.format(len(image_dir_content),
                                               len(filtered_image_dir_content),
                                               len(counter_removed_no_bboxes),
                                               len(counter_removed_invalid_class)))

    """
    Plotting some random samples form removed and valid images.
    """

    title = 'Random Selection of removed images without bounding boxes:'
    plot_sample_images_dataset(image_dir, title, counter_removed_no_bboxes)

    title = 'Random Selection of removed images with invalid class:'
    plot_sample_images_dataset(image_dir, title, counter_removed_invalid_class)

    title = 'Random Selection of filtered dataset:'
    plot_sample_images_dataset(image_dir, title, filtered_image_dir_content)

    # Return a list of valid images.
    return filtered_image_dir_content


def load_classname_2_classid_dict(raw_open_images_dir):
    """
    This function is loading a dictionary that maps from the OpenImages classnames to the respective class_ids.
    :param raw_open_images_dir: The path to the raw OpenImages images.
    :return: Returns a dictionary that maps from the image class_names to the respective class_ids.
    """

    # Init the dictionary.
    classname_2_classid = bidict()

    # Open the class description file from OpenImages.
    with open(raw_open_images_dir + '/class-descriptions-boxable.csv', 'r') as file:
        for line in file:
            # Extract line info.
            classid, classname = line.replace('\n', '').split(',')
            # Save to dict.
            classname_2_classid[classname] = classid

    # Return dictionary.
    return classname_2_classid


def load_imageid_2_bboxes_dict(raw_open_images_dir):
    """
    This function load a dictionary that maps from the OpenImages files to the respective containing bounding boxes.
    :param raw_open_images_dir: The path to the raw OpenImages images.

    :return: Returns a dictionary that maps from the image names to the containing bounding boxes.
    """
    # Init the Default Dict.
    imageid_2_bboxes = defaultdict(list)

    # The files containing the bounding boxes and annotations.
    annotations_bbox_files = ['test-annotations-bbox.csv', 'validation-annotations-bbox.csv']

    for annotations_bbox_file in annotations_bbox_files:

        # Open the annotation file.
        with open(raw_open_images_dir + '/' + annotations_bbox_file, 'r') as file:
            # Go through the annotation files.
            for line_id, line in enumerate(file):
                # Skip header.
                if line_id == 0:
                    continue

                # Extract the interesting part of the annotation line.
                img_id, _, class_id, _, x_min, x_max, y_min, y_max, _, _, _, _, _ = line.replace('\n', '').split(',')

                # Save the classes and box coordinates in the list.
                imageid_2_bboxes[img_id].append((class_id, x_min, x_max, y_min, y_max))

    # Return the dict to user.
    return imageid_2_bboxes


def plot_sample_images_dataset(image_dir, plt_title, image_names=None):
    """
    This function is used to plot a random sample of images out of a image_dir. It is used to
    showcase the data to the user.
    Args:
        plt_title: If given the title of the plot shown.
        image_dir: The directory scraped for images.
        image_names: The names of the images in directory to choose from. If None all images used.

    Returns: None, but plots the samples in grid fashion.

    """
    # In case no names are defined get the full content of the dir.
    if not image_names:
        image_names = os.listdir(image_dir)

    # Sample a random list of images.
    random_sample_valid_images = np.random.choice(image_names, 64)

    # Load the images and put the in a shared list.
    images = []
    for image_name in random_sample_valid_images:
        images.append(mpimg.imread(image_dir + '/' + image_name))

    # Create the Figure and the image gird.
    fig = plt.figure(figsize=(9, 9), num=plt_title)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(8, 8),
                     axes_pad=0.1,
                     )

    # Plot the individual images.
    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.axis('off')

    # Set the title if needed.
    fig.suptitle(plt_title, y=0.92)

    # Show the plots.
    plt.show()
    # Draw the figure.
    fig.canvas.draw()


def crop_bbox_resize_save(raw_open_images_dir, image_names, img_dim=(64, 64)):
    """
    This function extract a randomly chosen bounding box our of the image and crops it to a square of maximum size
    that is the resized to the wished resolution.
    :param img_dim: The final image dimension for resizing.
    :param raw_open_images_dir: The path to the raw OpenImages images.
    :param image_names: The names of raw OpenImages to process.
    :return: A dictionary mapping from image classes to images created in the respective class. Also writes new images.
    """

    # The directory holding the raw images.
    image_dir = raw_open_images_dir + '/raw_open_images'

    # The directory holding the new cropped and resized images.
    new_image_dir = raw_open_images_dir + '/cropped_and_resized_images'
    Path(new_image_dir).mkdir(exist_ok=True)

    # Load the mapping from image names to their bounding boxes.
    imageid_2_bboxes = load_imageid_2_bboxes_dict(raw_open_images_dir)
    # Load the mapping from classname to classids.
    classname_2_classid = load_classname_2_classid_dict(raw_open_images_dir)

    # A helper variable to plot some showcases for data creation.
    num_showcases = len(image_names) / 10

    # The dictionary returned for mapping.
    img_2_label = defaultdict(list)

    # Going thorough all the images.
    for image_indx, image_name in enumerate(tqdm(image_names)):

        # Extracting the image id.
        image_id = image_name.replace('.jpg', '')

        # Extracting bounding boxes in image.
        bounding_boxes_image = imageid_2_bboxes[image_id]

        # Choose a random bounding box.
        random_bbox_index = np.random.choice(range(len(bounding_boxes_image)), 1)[0]

        # Get the chosen bounding box.
        random_bbox = bounding_boxes_image[random_bbox_index]
        # Extract bounding box location.
        bbox_xy_norm = random_bbox[1:]
        bbox_xy_norm = [float(i) for i in bbox_xy_norm]
        # Extract bbox label.
        bbox_label = classname_2_classid.inverse[random_bbox[0]]

        # Load the respective image.
        img = Image.open(image_dir + '/' + image_name)

        # Get size of image.
        width, height = img.size

        # Calculate Crop pixel coordinate.
        x_min = int(width * bbox_xy_norm[0])
        x_max = int(width * bbox_xy_norm[1])
        y_min = int(height * bbox_xy_norm[2])
        y_max = int(height * bbox_xy_norm[3])
        # Combine them in a cropping tuple.
        bbox_xy = (x_min, y_min, x_max, y_max)

        # Crop the image.
        cropped_img = img.crop(bbox_xy)

        # Get size of cropped image.
        width, height = cropped_img.size

        '''
        Now we need to make sure that we have an square image before we resize to our square target, because we
        don't want to alter the aspect ratio. This is done by cropping a randim square with maximum size out of the 
        bounding box.
        '''

        # For width smaller height keep width constant and crop height randomly.
        if width < height:
            # The maximum allowed start point for cropped dimension.
            max_allowed_y = height - width
            # Get a random crop starting point.
            y_min = np.random.randint(low=0, high=max_allowed_y, size=1)[0]
            # Calculate crop end point.
            y_max = y_min + width
            # Define crop box.
            crop_box = (0, y_min, width, y_max)
            # Crop image.
            cropped_img_square = cropped_img.crop(crop_box)

        # For width larger height keep height constant and crop width randomly.
        elif width > height:
            # The maximum allowed start point for cropped dimension.
            max_allowed_x = width - height
            # Get a random crop starting point.
            x_min = np.random.randint(low=0, high=max_allowed_x, size=1)[0]
            # Calculate crop end point.
            x_max = x_min + height
            # Define crop box.
            crop_box = (x_min, 0, x_max, height)
            # Crop image.
            cropped_img_square = cropped_img.crop(crop_box)

        # If it is already square we do nothing.
        else:
            cropped_img_square = cropped_img

        # Assert that we know have a square.
        width, height = cropped_img_square.size
        assert width == height

        # Resize image to img_size using bicubic interpolation.
        cropped_and_resized_img = cropped_img_square.resize(img_dim, Image.BICUBIC)

        # Save the new cropped and resized image to disk.
        cropped_and_resized_img.save(new_image_dir + '/' + image_name)

        # Link the label of the bbox cropped with the image name.
        img_2_label[bbox_label].append(image_name)

        '''
        From time to time we show the user an example of the dataset creation so that he can understand how 
        counterexamples are created.
        '''
        if image_indx % num_showcases == 0:
            # The set of images to plot.
            images = [img, cropped_img, cropped_img_square, cropped_and_resized_img]

            # The figure.
            fig, ax = plt.subplots(1, 4, figsize=(15, 4))
            axes = ax.flatten()

            plt.suptitle('Showing processing example for class {}'.format(bbox_label), fontsize=18, y=1.05)

            image_titles = ['original', 'bounding box', 'max size random square', 'counter example']

            # Add plots to figure.
            for img_id, img in enumerate(images):
                axes[img_id].imshow(img)
                axes[img_id].axis('off')
                axes[img_id].title.set_text(image_titles[img_id])

            # Show the plots.
            plt.show()
            # Draw the figure.
            fig.canvas.draw()

    # Return the dict mapping image names to labels.
    return img_2_label


def split_train_test(raw_open_images_dir, preprocessed_images_2_label, split_ratio=0.1212):
    """
    This function splits the OpenImages in a train and validation set, based on a provided ration.
    The default is chosen so that the result is as close to the FairFace split as possible. Splitting is done on
    classes so that the overall distribution stays same. Extraction per class is random.
    Args:
        raw_open_images_dir: The path to the raw OpenImages images.
        preprocessed_images_2_label: The dataset directory mapping from a class to all respective images.
        split_ratio: Ratio to split number of samples per class.

    Returns: None, but writes csv files to disk.

    """
    # Lists holding the training and test samples.
    training_data = []
    test_data = []

    print('Splitting dataset and writing to csv files in {}...'.format(raw_open_images_dir))

    # For every class in the dataset.
    for image_class, images_in_class in tqdm(preprocessed_images_2_label.items()):

        # Calculate number of test samples to extract.
        num_images_in_class = len(images_in_class)
        num_images_in_class_test = int(num_images_in_class * split_ratio)

        # Get a set of random samples that will be moved to testset.
        test_samples_indx = np.random.randint(0, num_images_in_class, num_images_in_class_test)

        # Go through samples and either put in test or train based on indices drawn.
        for image_indx, image in enumerate(images_in_class):
            # Save image name and class to respective dataset.
            if image_indx in test_samples_indx:
                test_data.append((image, image_class))
            else:
                training_data.append([image, image_class])

    '''
    Write the split to files
    '''

    # Write the train split.
    with open(raw_open_images_dir + '/openimages_label_train.csv', mode='w') as train_file:
        # Define CSV writer.
        csv_writer = csv.writer(train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write header.
        csv_writer.writerow(['Image Filename', 'Image Class'])

        for sample in training_data:
            # Write sample info.
            csv_writer.writerow(sample)

    # Write the test split.
    with open(raw_open_images_dir + '/openimages_label_val.csv', mode='w') as test_file:
        # Define CSV writer.
        csv_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(['Image Filename', 'Image Class'])

        for sample in test_data:
            # Write sample info.
            csv_writer.writerow(sample)


def resize_fair_face_images(raw_fair_face_dir, img_size=(64, 64)):
    """
    This function resizes the FairFace dataset to our needed image size inplace.
    Args:
        img_size: The new image size to reshape to.
        raw_fair_face_dir: The path to the raw FairFace images.

    Returns: None, but alters files on disk and prints examples of process.

    """

    # Extracts the downloaded zip file to directory.
    with zipfile.ZipFile(raw_fair_face_dir + '/fairface-img-margin025-trainval.zip', 'r') as zip_ref:
        zip_ref.extractall(raw_fair_face_dir)

    title = "Random samples of FairFace dataset before resizing..."
    image_dir = raw_fair_face_dir + '/val'
    # Prints some random samples for user.
    plot_sample_images_dataset(image_dir, title)

    # The two dataset types
    folders = ['/train', '/val']

    for folder in folders:
        # The respective subfolder of the images.
        images_dir = raw_fair_face_dir + folder

        # Get all samples in that folder.
        dir_content = os.listdir(images_dir)

        print("Resizing FairFace Images in {}...".format(folder))

        for image_name in tqdm(dir_content):
            # Load the image.
            img = Image.open(images_dir + '/' + image_name)

            # Check size of image to ensure it is square.
            width, height = img.size
            assert width == height

            # Resize image to img_size using bicubic interpolation.
            resized_img = img.resize(img_size, Image.BICUBIC)

            # Overwrite image with resized version.
            resized_img.save(images_dir + '/' + image_name)

    title = "Random samples of FairFace dataset after resizing..."
    # Prints some random samples for user.
    plot_sample_images_dataset(image_dir, title)


def transfer_fairface_images(raw_fair_face_dir, our_dataset_dir, remove_source=False):
    """
    This function copies the FairFace image folders to our new dataset. This can be done
    on folder level as they are already split.
    Args:
        raw_fair_face_dir: The source folder containing our preprocessed OpenImages.
        our_dataset_dir: The path of our new created dataset.
        remove_source: If true the source folders will get deleted.

    Returns: None, but alters files on disk.

    """

    # The two types of datasets to write.
    folders = ['/train', '/val']

    for folder in folders:
        print("Transferring FairFace {} images...".format(folder.replace('/', '')))

        # Copies image folders to new destination.
        folder_to_copy_from = raw_fair_face_dir + folder
        folder_to_copy_to = our_dataset_dir + folder
        copy_tree(folder_to_copy_from, folder_to_copy_to)

        # Removes source data if flag set.
        if remove_source:
            shutil.rmtree(folder_to_copy_from)
        print('    DONE!')


def transfer_open_images_images(raw_open_images_dir, our_dataset_dir):
    """
    This function transfers the preprocessed OpenImages images to their
    respective train or validation folder in our new dataset folder based on the split
    executed before.
    Args:
        raw_open_images_dir: The source folder containing our preprocessed OpenImages.
        our_dataset_dir: The path of our new created dataset.

    Returns: None, but alters files on disk.

    """

    # The two types of datasets to write.
    dataset_types = ['train', 'val']

    for dataset_type in dataset_types:

        # Read dataset label file created by us during split.
        with open(raw_open_images_dir + '/openimages_label_' + dataset_type + '.csv', mode='r') as label_file:
            print("Transferring OpenImages {} images...".format(dataset_type))

            for line_id, line in enumerate(label_file):
                # Skip header.
                if line_id == 0:
                    continue

                # Extract filename
                img_file_name, _ = line.replace('\n', '').split(',')

                # Copy file from src folder to new dst folder.
                src = raw_open_images_dir + '/cropped_and_resized_images/' + img_file_name
                dst = our_dataset_dir + '/' + dataset_type + '/' + img_file_name
                shutil.copyfile(src, dst)

            print('    DONE!')


def create_label_csv(raw_fair_face_dir, raw_open_images_dir, our_dataset_dir):
    """
    This function writes the final csv label files for our created dataset, by combining the individual of
    FairFace and OpenImages.
    Args:
        raw_fair_face_dir: The path to the raw FairFace data.
        raw_open_images_dir: The path to the raw OpenImages data.
        our_dataset_dir:  The path of our new created dataset.

    Returns: None, but writes files to disk.

    """

    # The two types of datasets to write.
    dataset_types = ['train', 'val']

    for dataset_type in dataset_types:
        print("Transferring {} labels...".format(dataset_type))

        # Opening the label file of our dataset.
        with open(our_dataset_dir + '/our_dataset_label_' + dataset_type + '.csv', mode='w') as our_label_file:
            # Define a csv wirter.
            csv_writer = csv.writer(our_label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Write header.
            csv_writer.writerow(['image_filename', 'age', 'gender', 'race', 'image_class'])

            # Reading the FairFace label file and transfer.
            with open(raw_fair_face_dir + '/fairface_label_' + dataset_type + '.csv', mode='r') as label_file:

                print("    Transferring FairFace {} labels...".format(dataset_type))
                for line_id, line in enumerate(label_file):
                    # Skip header.
                    if line_id == 0:
                        continue

                    # Extract line info.
                    image_filename, age, gender, race, _ = line.replace('\n', '').split(',')
                    # Write subset of info in our new label style and add class name.
                    csv_writer.writerow([image_filename, age, gender, race, 'Human Face'])
                print('        DONE!')

            # Reading the OpenImages label file and transfer.
            with open(raw_open_images_dir + '/openimages_label_' + dataset_type + '.csv', mode='r') as label_file:

                print("    Transferring OpenImages {} labels...".format(dataset_type))
                for line_id, line in enumerate(label_file):
                    # Skip header.
                    if line_id == 0:
                        continue

                    # Extract line info.
                    image_filename, image_class = line.replace('\n', '').split(',')
                    # Write subset of info in our new label style and skip non present info.
                    csv_writer.writerow([dataset_type + '/' + image_filename, "", "", "", image_class])
                print('        DONE!')


def reduce_dataset_size_keep_class_dist(preprocessed_images_2_label_all, new_size=97698):
    """
    This function reduces the size of the filtered dataset given in preprocessed_images_2_label_all
    to new_size. It first removes an equal percentage of samples on every class to get as close to the
    final size as possible while keeping the class distribution the same. The remaining samples are removed by
    uniform random choice over the whole dataset.
    Args:
        preprocessed_images_2_label_all: dataset dictionary containing all samples
        new_size: the size the dataset should be reduced to (default is size of FairFace dataset)

    Returns: An updated dataset dictionary containing new_size samples.

    """

    # Create a copy of the dict so we don not alter inplace of the passed one.
    preprocessed_images_2_label = copy.deepcopy(preprocessed_images_2_label_all)

    # Count images in dataset.
    number_of_valid_images = 0
    for _, images_in_class in preprocessed_images_2_label.items():
        number_of_valid_images += len(images_in_class)

    # Assert we can actually remove samples.
    assert number_of_valid_images > new_size

    print('Reducing dataset from {} to {} samples...'.format(number_of_valid_images, new_size))

    '''
    Removing the first samples equally over classes.
    '''

    # That is the percentage of samples the shall be removed per class.
    ratio_reduce_by_class = (number_of_valid_images - new_size) / number_of_valid_images

    # Go over the classes.
    for image_class, images_in_class in tqdm(preprocessed_images_2_label.items()):

        # Calculate the number of images to remove.
        num_images_in_class = len(images_in_class)
        num_images_in_class_to_remove = math.floor(num_images_in_class * ratio_reduce_by_class)

        # Choose a random subset of samples to remove.
        remove_samples_indx = np.random.randint(0, num_images_in_class, num_images_in_class_to_remove)

        # Remove the chosen samples.
        for image_indx, image in enumerate(images_in_class):
            if image_indx in remove_samples_indx:
                del images_in_class[image_indx]

    # Recount dataset size.
    number_of_valid_images = 0
    for _, images_in_class in preprocessed_images_2_label.items():
        number_of_valid_images += len(images_in_class)

    assert number_of_valid_images > new_size

    '''
    Removing the last remaining samples per random choice.
    '''

    # Calculate the number of remaining samples to remove.
    rest_to_remove = number_of_valid_images - new_size
    # Get classes in dataset.
    img_classes = list(preprocessed_images_2_label.keys())

    # Until we have removed the remaining samples.
    images_removed = 0
    while images_removed != rest_to_remove:
        # Select a random class to delete a sample.
        remove_class = np.random.choice(img_classes)
        images_in_class = preprocessed_images_2_label[remove_class]

        # Select a random sample in that class.
        num_images_in_class = len(images_in_class)
        # We keep a minimum of samples in classes.
        if num_images_in_class > 1000:
            remove_sample_indx = np.random.randint(0, num_images_in_class, 1)[0]
            images_removed += 1

            # Remove that sample.
            del images_in_class[remove_sample_indx]

    # Recount samples.
    number_of_valid_images = 0
    for _, images_in_class in preprocessed_images_2_label.items():
        number_of_valid_images += len(images_in_class)

    # Assert we reached our target.
    assert number_of_valid_images == new_size

    # Return new dictionary.
    return preprocessed_images_2_label
