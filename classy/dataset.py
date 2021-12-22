from torch.utils.data.dataset import Dataset
import os, sys
from skimage import io, color
import numpy as np
# from matplotlib import pyplot as plt
import torch
from torchvision import transforms


############ dataloader ############
result_classes = {
    0: 'no_THA',
    1: 'yes_THA',
    # 2: 'yes_HRA'
}

dataset_dir = 'dataset'
directories = {}
for class_num in result_classes:
    directories['train_' + str(class_num)] = os.path.join('train', result_classes[class_num])
    directories['val_' + str(class_num)] = os.path.join('val', result_classes[class_num])
    directories['test_' + str(class_num)] = os.path.join('test', result_classes[class_num])

# directories = {
#     'train_0' : 'no_THA_train',
#     'train_1' : 'yes_THA_train',
#     'val_0' : 'no_THA_val',
#     'val_1' : 'yes_THA_val',
#     'test_0' : 'no_THA_test',
#     'test_1' : 'yes_THA_test'
# }


class read_dataset(Dataset):
    def __init__(self, mode, transform=None):
        """
        Args:
           transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.sample_paths = []
        self._init(mode)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path,label = self.sample_paths[idx]
    
        x = io.imread(img_path)
    
        # in order to make everything in RGB (x, y, 3) dimension
        shape = x.shape
        if len(shape) == 3 and shape[2] > 3: # for cases (x, y, 4)
            x = x[:,:,:3]  # for JHU scoliosis images, originally 3 which is blank
            x = color.gray2rgb(x)
        elif len(shape) == 2: # for cases (x, y)
            x = color.gray2rgb(x)

        # square crop from bottom
        # ydim = shape[1]
        # xmax = shape[0]
        # xmin = xmax - ydim
        # x = x[xmin:xmax, :, :]

        # load in with values of range [0, 1]
        # per channel:
        # for idx in range(3):
            # first get rid of negative values by adding min to all values - if min=0, great
            # actually, it doesn't look like there are any negatives
            # imgmin = np.amin(x[:,:,idx])
            # then divide all values by max + min to get [0, 1]
            # imgmax = np.amax(x[:,:,idx])
            # x[:,:,idx] = x[:,:,idx] / imgmax
        
        # debug: check if it worked
        # print('dataloader: img min is:',np.amin(x))
        # print('dataloder: img max is:',np.amax(x))
 
        # in order to make sure we have all images in uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)

        if self.transform:
            x = self.transform(x)

        def normalize_img(img_tensor):
            img_std, img_mean = torch.std_mean(img_tensor)
        
            transform = transforms.Compose([
                transforms.Normalize(img_mean, img_std)]) 
        
            normalized_img = transform(img_tensor)
        
            return normalized_img
        
        x = normalize_img(x)
    
        return (x, label, img_path)

    def _init(self, mode):
        subdir = {}

        # Result class iteration
        for class_num in result_classes:
            subdir[class_num] = os.path.join(dataset_dir, directories[mode + '_' + str(class_num)])
            samples = os.listdir(subdir[class_num])
            for sample in samples:
                if not sample.startswith('.'):  # avoid .DS_Store
                    img_path = os.path.join(subdir[class_num], sample)
                    self.sample_paths.append((img_path, class_num))


if __name__ == '__main__':
    print('Test codes are commented out')
