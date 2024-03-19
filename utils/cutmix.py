import numpy as np
import torch
from torchvision.transforms import v2 

def rand_bbox(size):
        
    lam = np.random.beta(1., 1.)
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.intc(W * cut_rat)
    cut_h = np.intc(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(images,labels):
    tr=v2.RandomPhotometricDistort(p=1)
    if np.random.rand(1) < 0.5:
        rand_index = torch.randperm(images.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size())
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
        images = tr(images)
    return images,labels
    

