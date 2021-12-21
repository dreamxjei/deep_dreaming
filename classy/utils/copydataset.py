# Idea: Take source and new image directories as arguments.
# Move all file names from first directory to second directory, etc.
# Repeat for second group of files, etc.

import os
import pathlib
import numpy as np
from shutil import copyfile
from random import shuffle

result_classes = {
    0: 'no',
    1: 'yes'
    }

sourcedir = 'risser_data'
split = [0.7, 0.1, 0.2]  # train val test split ratios

dataset_dir = 'dataset'
directories = {}
for class_num in result_classes:
    iter_class = result_classes[class_num]
    directories['train_' + str(class_num)] = os.path.join('train', iter_class)
    directories['val_' + str(class_num)] = os.path.join('val', iter_class)
    directories['test_' + str(class_num)] = os.path.join('test', iter_class)
    pathlib.Path(os.path.join(dataset_dir, directories['train_' + str(class_num)])).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(dataset_dir, directories['val_' + str(class_num)])).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(dataset_dir, directories['test_' + str(class_num)])).mkdir(parents=True, exist_ok=True)

    # in order: sourcedir, traindir, valdir, testdir, trainnum, valnum, testnum
    iter_class = result_classes[class_num]
    dir1 = os.path.join(sourcedir, str(iter_class))
    dir2 = os.path.join(dataset_dir, 'train', str(iter_class))
    dir3 = os.path.join(dataset_dir, 'val', str(iter_class))
    dir4 = os.path.join(dataset_dir, 'test', str(iter_class))
    # num1 = int(input("Number of training files: "))
    # num2 = int(input("Number of validation files: "))
    # num3 = int(input("Number of test files: "))

    onlyfiles = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    
    # randomize
    shuffle(onlyfiles)
    
    # automatically sort train-val-test w/ given ratios
    numfiles = len(onlyfiles)
    num1 = np.uint8(np.floor(numfiles * split[0]))
    # num2 = numfiles * split[1]
    num3 = np.uint8(np.floor(numfiles * split[2]))
    num2 = np.uint8(numfiles - num1 - num3)
    
    for i in range(num1):
    	copyfile(dir1 + "/" + onlyfiles[i], dir2 + "/" + onlyfiles[i])
    
    for i in range(num1, num1 + num2):
    	copyfile(dir1 + "/" + onlyfiles[i], dir3 + "/" + onlyfiles[i])
    
    for i in range(num1 + num2, num1 + num2 + num3):
    	copyfile(dir1 + "/" + onlyfiles[i], dir4 + "/" + onlyfiles[i])
    
