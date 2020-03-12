#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder 
    and dump them in a results folder """

import torch

from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
from models.vgg import *
from models.resnet import *
from misc_functions import *
import torch.nn.functional as F

# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/pixelperturbation/'

batch_size = 5
total_pixels = 224*224

cuda = torch.cuda.is_available()
device = torch.device("cpu") # no training, cpu should be sufficient

# Dataset loader for sample images
sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((224,224)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

transform_image = transforms.ToPILImage()
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])


# uncomment to use VGG
model = vgg16_bn(pretrained=True).to(device)
# model = resnet18(pretrained=True).to(device)

# Initialize FullGrad objects
fullgrad = FullGrad(model)
simple_fullgrad = SimpleFullGrad(model)

save_path = PATH + 'results/'

def return_k_index_argsort(img, k, method):
    """
        Returns a 1d array of the indices - ranked accordingly.
    """
    idx = np.argsort(img.flatten())

    if (method == "fullgrad") or (method == "inputgrad") or (method == "simplegrad"):
        return np.column_stack(np.unravel_index(idx[::-1][k:], img.shape))

    elif method == "random":
        k = total_pixels - k
        idx = np.random.choice(idx.shape[0], k, replace=True)
        return np.column_stack(np.unravel_index(idx, img.shape))


def replace_pixels(img, idx, approach = 'zero'):
    """
        Given a set of indices, one image and an approach this function perburtates
        specific pixels and sets the value accordingly.
    """
    if approach == 'zero':
        for x,y in idx:
            img[:,x,y] = 0.0
    elif approach == 'mean':
        mean_r, mean_g, mean_b = calc_mean_channels(img)
        for x,y in idx:
            img[0,x,y] = mean_r
            img[1,x,y] = mean_g
            img[2,x,y] = mean_b
    return img

def show_sal_scores(idx, sal_map):
    """
        Prints the salience score stored in the saliency map
    """
    for i in idx:
        print(f'coords: {i} gives score {sal_map[i[0]][i[1]]}')

def compute_saliency_and_save(k, method):
    """
        Computes the saliency map for a given picture and returns a stack of
        former output vectors and a stack of pixel perturbated images.
    """
    former_outputs, new_images_to_forward, image_counter = [], [], 0

    image_counter = 0
    max_iter = 5
    for batch_idx, (data, target) in enumerate(sample_loader):
        data, target = data.to(device).requires_grad_(), target.to(device)
        if method == "inputgrad" or method == "random" or method == "fullgrad":
            inputgrad_cam, cam, model_output = fullgrad.saliency(data)
            if method == "inputgrad":
            # Compute saliency maps for the input data.
                cam = inputgrad_cam
        elif method == "simplegrad":
            cam, model_output = simple_fullgrad.saliency(data)
        # Find most important pixels and replace.
        for i in range(data.size(0)):
            # Append former output to a tensor.
            former_outputs.append(model_output[i])   

            # Get unnormalized image and heat map.
            sal_map = cam[i,:,:,:].squeeze()
            image_to_save = unnormalize(data[i,:,:,:]) #unnormalize to show output
            image_to_forward = data[i,:,:,:] #dont unnormalize but keep normalized for model.forward()
            # Get k indices and replace within image
            indices = return_k_index_argsort(sal_map.detach().numpy(), k, method)
            new_image_to_save = replace_pixels(image_to_save, indices, 'zero')
            new_image_to_forward = replace_pixels(image_to_forward, indices, 'zero')
            new_images_to_forward.append(new_image_to_forward)

            curr_perc = 100 - round(k/total_pixels,2)*100
            print(f"Current image counter ID: {image_counter}")
            
            image_counter += 1
        if image_counter > max_iter:
            break
    image_counter = 0

    return torch.stack(former_outputs), torch.stack(new_images_to_forward)


def compute_pertubation(k, experiment, method = 'pp'):
    """
        Computes the pixel perturbation for a given experiment and method.
        Returns both the result for a given k as the results needed to calculate the standard deviation
    """
    # Get adjusted images and fetch former outputs
    former_outputs, new_images = compute_saliency_and_save(k, method)
    print("new images shape: ", new_images.shape)
    # Create new outputs
    new_model_output = model.forward(new_images)
    afoc_results = []
    # Calculate absolute fractional output change
    total = 0
    img_counter = 0
    for i, former_output in enumerate(former_outputs):
        new_output = new_model_output[i]
        new_output = torch.nn.functional.softmax(new_output.data)
        former_output = torch.nn.functional.softmax(former_output.data)

        if experiment == "AFOC":
            max_index = former_output.argmax()
            diff_den = abs(new_output[max_index]-former_output[max_index])
            print(f"old value: {former_output[max_index]} and new value: {new_output[max_index]}")
            afoc_results.append(diff_den/former_output[max_index])
            total += diff_den/former_output[max_index]

        elif experiment == "KL-divergence":
            kl_div = calc_kl_div(former_output, new_output)
            total += kl_div
            afoc_results.append(kl_div)
        img_counter += 1

    print(f"=== Current K = {100 - round(k/total_pixels,2)*100} for method = {method} on experiment {experiment} yields the following results ===")
    print(f"Total summed differences: {total}")
    print(f"We will divide by: {img_counter}")
    print(f"Which will become: {(total/img_counter)}")
    return (total/img_counter), np.asarray([x/img_counter for x in afoc_results])

def calc_kl_div(a,b):
    """
        Calculates the KL-divergence with ref tensor a, comparing to a new tensor b.
        source of formula: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    return (a * (a / b).log()).sum()

def obtain_percentages(method):
    """
        Returns a list of percentages that will later be used to decide how many pixels to remove.
    """
    if method == "fullgrad" or method == "random" or method == "inputgrad":
        percentages = [0.001, 0.005,0.01,0.03,0.05,0.07,0.1]
        #percentages = [0.01,0.05,0.1]
        #percentages = [0.001, 0.005]
        return percentages, [round(total_pixels - (k * total_pixels)) for k in percentages]

    elif method == "roar":
        percentages = [0.1, 0.3, 0.5,0.7,0.9]
        return [round(total_(k * total_pixels)) for k in percentages]


def pixel_pertubation(experiment):
    """
        Messy big function to obtain results and plot them
        Takes the experiment "KL-divergence" or "AFOC" as input and plots the results.
    """
    method = "inputgrad"
    percentages, Ks = obtain_percentages(method)
    results_IG, results_R, results_FG, results_simple_FG = [],[],[], []
    results_STD_IG, results_STD_R, results_STD_FG, results_STD_simple_FG = [], [],[],[]
    for k_index, k in enumerate(Ks):
        result_random, afoc_random = compute_pertubation(k, experiment, method = "random")
        result_inputgrad, afoc_inputgrad = compute_pertubation(k, experiment, method = "inputgrad")
        result_fullgrad, afoc_fullgrad = compute_pertubation(k,experiment,method = "fullgrad")
        #result_simple_FG, afoc_simplegrad = compute_pertubation(k, experiment, method = "simplegrad")
        
        results_R.append(result_random)
        results_IG.append(result_inputgrad)
        results_FG.append(result_fullgrad)
        #results_simple_FG.append(result_simple_FG)

        results_STD_R.append(np.std(afoc_random))
        results_STD_IG.append(np.std(afoc_inputgrad))
        #results_STD_simple_FG.append(np.std(afoc_simplegrad))
        results_STD_FG.append(np.std(afoc_fullgrad))


    print(f'percentages: {percentages}')
    print(f'results for fullgrad: {results_FG}')
    print(f'results for inputgrad: {results_IG}')
    print(f'results for random: {results_R}')
    #print(f'results for simple fullgrad: {results_simple_FG}')

    print_items(results_STD_R, "Random")
    print_items(results_STD_IG, "InputGrad")
    #print_items(results_STD_simple_FG, "SimpleFullGrad")
    print_items(results_STD_FG, "FullGrad")

    plt.errorbar(percentages, results_R, results_STD_R, marker='o', label="Random")
    plt.errorbar(percentages, results_IG, results_STD_IG, marker = 'o', label = "InputGrad" )
    #plt.errorbar(percentages, results_simple_FG, results_STD_simple_FG, marker = 'o', label = "Simple FullGrad")
    plt.errorbar(percentages, results_FG, results_STD_FG, marker = 'o', label = "FullGrad")

    plt.xlabel("Percentages")
    plt.xscale("log")
    if experiment == "AFOC":
        plt.ylabel("Absolute Fractional Output Change")
        
    elif experiment == "KL-divergence":
        plt.ylabel("KL-divergence value")

    plt.title(f"Removing % least salient pixels vs {experiment}")
    plt.legend()
    plt.show()


def print_items(items, method):
    """
        Prints the results in the terminal
    """
    print(f"Method {method} has the following STD results {items}")

if __name__ == "__main__":
    # Create folder to saliency maps
    experiment = "AFOC"
    #experiment = "KL-divergence"
    create_folder(save_path)
    pixel_pertubation(experiment)



