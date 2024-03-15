from operator import add
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utils.one_hot_encode import one_hot,label_encode
from utils.data_loader import loader
from utils.Loss import Dice_CE_Loss

from models.Unet import UNET
# from models.Unet_sep import UNET_sep
# from models.Unet_Wavelet import UNET_wave

# from models.Unet_ViT_m import UNET_ViT_m
# from models.Unet_ViT_Wavelet_m import UNET_ViT_Wavenet_m
# from models.transunet import TransUNet_copy
# from models.UNETR import UNET_TR
# from models.ViT_segm import ViT_M

from models.CA_CBA_Unet import CA_CBA_UNET
from models.CA_Former_Unet import CA_Former_Unet
from models.CA_CBA_Convnext import CA_CBA_Convnext
from models.CA_Former_Convnext import CA_Former_Convnext

from models.Conv_Former_Unet import Conv_Former_Unet
from models.Conv_CBA_Convnext import Conv_CBA_Convnext

# from models.MetaPolyp import MetaPolyp
# from models.MetaFusion2 import Metafusion2
# from models.MetaFusion3 import Metafusion3
# from models.UNET_Convformer import UNET_Convformer
def IoU(y_true, y_pred):
    
    intersection = (y_true * y_pred).sum()
    union = (y_true).sum() + (y_pred).sum() - intersection
    
    return torch.mean( (intersection + torch.tensor(1e-6)) / (union +  torch.tensor(1e-6)))

def calculate_metrics(y_true, y_pred):

    y_true = y_true.numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard   = jaccard_score(y_true, y_pred)
    score_f1        = f1_score(y_true, y_pred)
    score_recall    = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc       = accuracy_score(y_true, y_pred)
    score_IoU       = IoU(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_IoU]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


if __name__ == "__main__":

    test_x = sorted(glob("test/images/*"))
    test_y = sorted(glob("test/masks/*"))

    checkpoint_path = "modelsave/checkpoint.pth"
    n_classes   = 1
    batch_size  = 2
    num_workers = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader,test_loader = loader(batch_size,num_workers,shuffle=True)
    
    model0 = UNET(n_classes).to(device) 
    model1 = CA_CBA_UNET(n_classes).to(device)
    model2 = CA_Former_Unet(n_classes).to(device) 
    model3 = CA_CBA_Convnext(n_classes).to(device) 
    model4 = CA_Former_Convnext(n_classes).to(device)
    model5 = Conv_Former_Unet(n_classes).to(device)
    model6 = Conv_CBA_Convnext(n_classes).to(device)

    #model2 = UNET_Convformer(n_classes).to(device) 
    # model8 = ViT_M(img_dim=256,
    #             in_channels=3,
    #             patch_dim=16,
    #             embedding_dim=768,
    #             block_num=6,
    #             head_num=4,
    #             mlp_dim=1024).to(device) 
    # model3 = UNET_ViT_Wavenet_m(n_classes).to(device) 
    # model7 = MetaPolyp(n_classes).to(device)
    # model5 = TransUNet_copy(img_dim=128,in_channels=3,out_channels=128,head_num=4,mlp_dim=512,block_num=8,encoder_scale=16,class_num=1).to(device) #1.5  5.3
    # model7 = ViT_segm(images_dim=128,input_channel=3,token_dim=768, n_heads=4, mlp_layer_size=512, t_blocks=12, patch_size=8, classification=False).to(device)
    
    
    all_models=[model0,model1,model2,model3,model4,model5,model6]


    for idx,model in enumerate(all_models):
        if idx == 3:

            model.eval()
            print(f"Testing for Model{idx} = {model.__class__.__name__}")
            checkpoint_path = "modelsave/checkpoint_model_"+str(model.__class__.__name__)
            
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(checkpoint_path))
            else: 
                model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

            metrics_score = [ 0.0, 0.0, 0.0, 0.0, 0.0]

            for batch in tqdm(test_loader, desc=f"testing ", leave=False):
                images,labels   = batch                
                model_output    = model(images)

                with torch.no_grad():

                    model_output    = model(images)
                    prediction = torch.sigmoid(model_output)

                    if n_classes>1:
                        prediction = torch.argmax(prediction,dim=2)    #for multiclass_segmentation

                    else:


                        score = calculate_metrics(labels, prediction)
                        metrics_score = list(map(add, metrics_score, score))


                jaccard     = metrics_score[0]/len(test_loader)
                f1          = metrics_score[1]/len(test_loader)
                recall      = metrics_score[2]/len(test_loader)
                precision   = metrics_score[3]/len(test_loader)
                acc         = metrics_score[4]/len(test_loader)

                print(f" Jaccard (IoU): {jaccard:1.4f} - F1(Dice): {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")
                print(len(test_loader))
                
'''
#### to show current output of model 

prediction    = prediction[0]        ## (1, 512, 512)
prediction    = np.squeeze(prediction)     ## (512, 512)
prediction    = prediction > 0.5
prediction    = np.array(prediction, dtype=np.uint8)

im_test    = np.array(images[0]*255,dtype=int)
im_test    = np.transpose(im_test, (2,1,0))
label_test = np.array(labels[0]*255,dtype=int)
label_test = np.transpose(label_test)

prediction = np.transpose(prediction)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(im_test)
plt.subplot(1, 3, 2)
plt.imshow(label_test,cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(prediction,cmap='gray')
'''
