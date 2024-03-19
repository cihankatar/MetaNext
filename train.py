
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
import time as timer
import argparse
import numpy as np

from tqdm import tqdm, trange
from utils.one_hot_encode import one_hot,label_encode
from utils.data_loader import loader
from utils.Loss import Dice_CE_Loss
from utils.cutmix import cutmix
from models.CA_CBA_Convnext import CA_CBA_Convnext
from torchvision.transforms import v2 

# from models.Unet_sep import UNET_sep
# from models.CA_Former_Convnext import CA_Former_Convnext
# from models.Unet_Wavelet import UNET_wave
# from models.Unet_ViT_m import UNET_ViT_m
# from models.Unet_ViT_Wavelet_m import UNET_ViT_Wavenet_m
# from models.transunet import TransUNet_copy
# from models.UNETR import UNET_TR
# from models.ViT_segm import ViT_M
# from models.CA_CBA_Unet import CA_CBA_UNET
# from models.CA_Former_Unet import CA_Former_Unet
# from models.Conv_Former_Unet import Conv_Former_Unet
# from models.Conv_CBA_Convnext import Conv_CBA_Convnext
# from models.MetaPolyp import MetaPolyp
# from models.MetaFusion2 import Metafusion2
# from models.MetaFusion3 import Metafusion3
# from models.UNET_Convformer import UNET_Convformer

def main():
    
    parser = argparse.ArgumentParser(
        prog="segmentation",
        description="Training ViT" )
    
    parser.add_argument("-m", "--model", type=int)   # -m 0  || --model=0

    args=parser.parse_args()
    args.model=0
    
    #parser.add_argument("-b", "--batch", type=int)   # -m 0  || --model=0
    #parser.add_argument("-e", "--epochs", type=int)   # -m 0  || --model=0
    #parser.add_argument("-l", "--learningrate", type=float)   # -m 0  || --model=0
    #parser.add_argument("-n", "--numberofclasses", type=int)   # -m 0  || --model=0

    #args.batch=2
    #args.epochs=200
    #args.learningrate=0.0001
    #args.numberofclasses=1

    #n_classes   = args.numberofclasses
    #batch_size  = args.batch    
    #epochs      = args.epochs
    #l_r         = args.learningrate
    #num_workers = 2

    n_classes   = 1
    batch_size  = 8   
    epochs      = 200
    l_r         = 0.0001
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)

    model0 = CA_CBA_Convnext(n_classes).to(device) 
    
    #model0 = UNET(n_classes).to(device) 
    #model1 = CA_CBA_UNET(n_classes).to(device)
    #model2 = CA_Former_Unet(n_classes).to(device) 
    #model4 = CA_Former_Convnext(n_classes).to(device)
    #model5 = Conv_Former_Unet(n_classes).to(device)
    #model6 = Conv_CBA_Convnext(n_classes).to(device)
    #model2 = UNET_Convformer(n_classes).to(device) 
    #model3 = UNET_ViT_Wavenet_m(n_classes).to(device) 
    #model7 = MetaPolyp(n_classes).to(device)
    #model5 = TransUNet_copy(img_dim=128,in_channels=3,out_channels=128,head_num=4,mlp_dim=512,block_num=8,encoder_scale=16,class_num=1).to(device) #1.5  5.3
    #model7 = ViT_segm(images_dim=128,input_channel=3,token_dim=768, n_heads=4, mlp_layer_size=512, t_blocks=12, patch_size=8, classification=False).to(device)
    
    
    # all_models=[model0,model1,model2,model3,model4,model5,model6]


    all_models  = [model0]
    model       = all_models[args.model]
    idx         = args.model

    best_valid_loss = float("inf")
    print(f"TRAINING FOR MODEL{idx} = {model.__class__.__name__}")
    checkpoint_path = "modelsave/checkpoint_model_"+str(model.__class__.__name__)

    optimizer = Adam(model.parameters(), lr=l_r)
    loss_function = Dice_CE_Loss()
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0.00001, last_epoch=-1)

    for epoch in trange(epochs, desc="Training"):

        epoch_loss = 0.0
        model.train()
        
        for batch in train_loader:
            images,labels   = batch  
            images,labels   = images.to(device), labels.to(device)
            images,labels   = cutmix (images,labels)

            #start=timer.time()
            model_output    = model(images)
            
            if n_classes == 1:  
                model_output     = model_output
                train_loss       = loss_function.Dice_BCE_Loss(model_output, labels)

            else:
                model_output    = torch.transpose(model_output,1,3) 
                targets_f       = label_encode(labels) 
                train_loss      = loss_function.CE_loss(model_output, targets_f)


            epoch_loss     += train_loss.item() 
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            #end=timer.time()
            #print(end-start)

        epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Epoch loss for Model{idx} = {model.__class__.__name__} : {epoch_loss}")

        valid_loss = 0.0
        model.eval()

        with torch.no_grad():
            #for batch in tqdm(test_loader, desc=f" Epoch {epoch + 1} in validation", leave=False):

            for batch in (test_loader):
                images,labels   = batch  
                images,labels   = images.to(device), labels.to(device)   
                model_output    = model(images)
                loss            = loss_function.Dice_BCE_Loss(model_output, labels)
                valid_loss     += loss.item()
                
            valid_epoch_loss = valid_loss/len(test_loader)

        if valid_epoch_loss < best_valid_loss:

            print(f"previous val loss: {best_valid_loss:2.4f} new val loss: {valid_epoch_loss:2.4f}. Saving checkpoint: {checkpoint_path}")
            best_valid_loss = valid_epoch_loss
            torch.save(model.state_dict(), checkpoint_path)
        
        print(f'\n Model{idx} = {model.__class__.__name__} = training Loss: {epoch_loss:.3f}, val. Loss: {valid_epoch_loss:.3f}')

if __name__ == "__main__":
   main()

# image=images[0].permute(2,1,0)
# label=labels[0].permute(2,1,0)
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(label)
