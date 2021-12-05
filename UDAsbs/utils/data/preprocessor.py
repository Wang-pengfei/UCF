from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
import torch
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None,):
        super(Preprocessor, self).__init__()
        self.dataset = []#dataset
        for inds, item in enumerate(dataset):
            self.dataset.append(item+(inds,))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        items = self.dataset[index]
        fname, camid, inds =items[0],items[-2],items[-1]
        pids = []
        for i, pid in enumerate(items[1:-2]):
            pids.append(pid)

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return [img, fname]+ pids+[camid, inds]





