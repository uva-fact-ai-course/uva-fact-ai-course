

import torch

from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import os
import cv2

# Import saliency methods and models
from saliency.fullgrad import FullGrad
from saliency.simple_fullgrad import SimpleFullGrad
import torch.nn.functional as F

results_norm = [0.2474, 0.4218, 0.4691, 0.5307, 0.6018]
results_random= [0.2412, 0.3967, 0.4743, 0.5435, 0.5917]
percentages = [0.01, 0.03,0.05,0.07,0.1]
percentages = [p*100 for p in percentages]

plt.xlabel("% removal least salient pixels")
plt.ylabel("Absolute Fractional Output Change")
plt.title("AFOC of FullGrad vs Random")
plt.plot(percentages, results_norm, label = "FullGrad", marker='o')
plt.plot(percentages, results_random, label = "Random", marker='o')
plt.legend()
plt.show()