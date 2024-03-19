##IMPORT 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

def cutout(img,lbl, pad_size, replace):
    _, h, w = img.shape
    center_h, center_w = torch.randint(high=h, size=(1,)), torch.randint(high=w, size=(1,))
    low_h, high_h = torch.clamp(center_h-pad_size, 0, h).item(), torch.clamp(center_h+pad_size, 0, h).item()
    low_w, high_w = torch.clamp(center_w-pad_size, 0, w).item(), torch.clamp(center_w+pad_size, 0, w).item()
    cutout_img = img.clone()
    cutout_lbl = lbl.clone()
    cutout_img[:, low_h:high_h, low_w:high_w] = replace
    cutout_lbl[:, low_h:high_h, low_w:high_w] = replace
    return cutout_img,cutout_lbl


class Cutout(torch.nn.Module):

    def __init__(self, p, pad_size, replace=0):
        super().__init__()
        self.p = p
        self.pad_size = int(pad_size)
        self.replace = replace

    def forward(self, image,lbl):
        if torch.rand(1) < self.p:
            cutout_image,lbl = cutout(image,lbl, self.pad_size, self.replace)
            return cutout_image,lbl
        else:
            return image,lbl
        
    def __repr__(self):
        return self.__class__.__name__ + "(p={0}, pad_size={1})".format(self.p, self.pad_size)
    


class KVasir_dataset(Dataset):
    def __init__(self,train_path,mask_path,transforms=None): #
        super().__init__()
        self.train_path      = train_path
        self.mask_path       = mask_path
        self.tr    = transforms

    def __len__(self):
         return len(self.train_path)
    
    def __getitem__(self,index):        

        #if 'jpg' in self.image_dir_list:
            image = Image.open(self.train_path[index])
            image = np.array(image,dtype=float)
            image = image.astype(np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

            #if self.transforms is not None:
             #   image = self.transforms(image)
        
            #if 'jpg' in self.image_dir_list:
            mask = Image.open(self.mask_path[index]).convert('L')
            mask = np.array(mask,dtype=float)
            mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)

            both_images = torch.cat((image, mask),0)
            both_images=self.tr(both_images)
            image=both_images[0:3] 
            mask=both_images[3]
            mask = mask.unsqueeze(0)

            cut = Cutout(0.5,25)
            image,mask=cut(image,mask)
            image=image/255
            mask=mask/255.001
            
            return image , mask

import matplotlib.pyplot as plt

# image=image.permute(2,1,0)
# label=mask.permute(2,1,0)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(label)