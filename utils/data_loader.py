from torch.utils.data import DataLoader
from utils.Custom_Dataset import KVasir_dataset
#from Test_Train_Split import split_main
from glob import glob
from torchvision.transforms import v2 

def loader(batch_size,num_workers,shuffle):

    train_im_path   = "train/images"
    train_mask_path = "train/masks"
    test_im_path    = "val/images"
    test_mask_path  = "val/masks"

# if not os.path.exists(train_im_path):
    # split_main()

    transformations = v2.Compose([  v2.Resize([512,512],antialias=True),
                                    v2.RandomHorizontalFlip(p=0.5),
                                    v2.RandomVerticalFlip(p=0.5),
                                    v2.RandomRotation(degrees=(0, 90)),
                                    #transforms.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225)),                                
                                  ])
 
    transformations_test = v2.Compose([  v2.Resize([512,512],antialias=True),
                                    #v2.RandomHorizontalFlip(p=0.5),
                                    #v2.RandomVerticalFlip(p=0.5),
                                    #v2.RandomRotation(degrees=(0, 90)),
                                    #transforms.Normalize(mean=(0.400, 0.485, 0.456, 0.406), std=(0,222, 0.229, 0.224, 0.225)),                                
                                  ])
 
                                                         

    train_im_path   = sorted(glob('train/images/*'))
    train_mask_path = sorted(glob('train/masks/*'))

    test_im_path    = sorted(glob("val/images/*"))
    test_mask_path  = sorted(glob("val/masks/*"))

    data_train  = KVasir_dataset(train_im_path,train_mask_path, transformations)
    data_test   = KVasir_dataset(test_im_path, test_mask_path, transformations_test)

    train_loader = DataLoader(
        dataset     = data_train,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers
        )
    
    test_loader = DataLoader(
        dataset     = data_test,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers
        )
    
    return train_loader,test_loader

    
#loader()