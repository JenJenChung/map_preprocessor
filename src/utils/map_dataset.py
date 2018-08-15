import sys
import numpy as np
import tarfile
import torch
from torch.utils.data.dataset import Dataset


class MapDataset(Dataset):
    def __init__(self, filename):
        try:
            tar = tarfile.open(filename, 'r')
        except IOError as e:
            print('I/O error({0}): {1}: {2}'.format(e.errno, e.strerror, filename))
            sys.exit()

        self.__filename = filename
        self.__num_files = len(tar.getnames())
        self.__memberslist = tar.getmembers()

    def __getitem__(self, index):
        tar = tarfile.open(self.__filename, 'r')
        file = tar.extractfile(self.__memberslist[index])
        data = torch.load(file)
        return np.array(data)

    def __len__(self):
        return self.__num_files
