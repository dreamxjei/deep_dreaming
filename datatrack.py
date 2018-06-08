# data track by mxj for deep dreaming

# a separate file to make reading specific data folders easier

import os, sys
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage import io, color

# data directory

dataset_dir = 'dataset/100_20_30'
directories = {
    'no_train' : 'no_THA_train',
    'yes_train' : 'yes_THA_train',
    'no_val' : 'no_THA_val',
    'yes_val' : 'yes_THA_val',
    'no_test' : 'no_THA_test',
    'yes_test' : 'yes_THA_test'
}

result_classes = {
    0 : 'no_THA',
    1 : 'yes_THA'
}

def main():
    # lorem ipsum

if __name__ == '__main__':
    main()
