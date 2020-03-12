import torch
import numpy as np

def return_k_index_argsort(salience_map, k, method):
    idx = np.argsort(salience_map.ravel())
    if method == "full_grad" or method == "input_grad":
        return np.column_stack(np.unravel_index(idx[:-k-1:-1], salience_map.shape))
    elif method == "pp":
        return np.column_stack(np.unravel_index(idx[::-1][:k], salience_map.shape))
    elif method == "random":
        idx = np.random.choice(idx.shape[0], k, replace=False)
        return np.column_stack(np.unravel_index(idx, salience_map.shape))

def get_k_based_percentage(img, percentage):
    w, h = img.shape
    numb_pix = w*h
    return numb_pix * percentage

def calc_rgb_means(img):
    mean_r = torch.mean(img[0,:,:])
    mean_g = torch.mean(img[1,:,:])
    mean_b = torch.mean(img[2,:,:])

    return mean_r, mean_g, mean_b

def replace_pixels(img, idx, approach = 'zero'):
    if approach == 'zero':
        for x,y in idx:
            img[:,x,y] = 0
    elif approach == 'mean':
        mean_r, mean_g, mean_b = calc_rgb_means(img)
        for x,y in idx:
            img[0,x,y] = mean_r
            img[1,x,y] = mean_g
            img[2,x,y] = mean_b

    return img