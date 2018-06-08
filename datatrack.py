# data track by mxj

# a separate file to make reading specific data folders easier

import os, sys
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage import io, color


# data directory

dataset_dir = 'dataset/100_20_30'
directories = {
    'train_0' : 'no_THA_train',
    'train_1' : 'yes_THA_train',
    'val_0' : 'no_THA_val',
    'val_1' : 'yes_THA_val',
    'test_0' : 'no_THA_test',
    'test_1' : 'yes_THA_test'
}

result_classes = {
    0 : 'no_THA',
    1 : 'yes_THA'
}


class importData():
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

    def __getitem_(self, idx):
        img_path, label = self.sample_paths[idx]
        x = io.imread(img_path)

        # in order to make everything in RGB (x, y, 3) dimension
        shape = x.shape
        if len(shape) == 3 and shape[2] > 3:  # for cases (x, y, 4)
            x = x[:, :, :3]
        elif len(shape) == 2:  # for cases (x, y)
            x = color.gray2rgb(x)

        # in order to make sure images are read as uint8
        if x.dtype != np.uint8:
            x = x.astype(np.uint8)

        if self.transform:
            x = self.transform(x)

        return(x, label)

    def _init(self, mode):
        dir_0 = os.path.join(dataset_dir, directories[mode + '_0'])
        dir_1 = os.path.join(dataset_dir, directories[mode + '_1'])

        # Result class iteration
        for class_num in result_classes:
            samples = os.listdir('dir_' + str(class_num))
            for sample in samples:
                if not sample.startswith('.'):  # avoid .DS_Store
                    img_path = os.path.join('dir_' + str(class_num), sample)
                    self.sample_paths.append((img_path, class_num))



if __name__ == '__main__':
    print('Main called')
