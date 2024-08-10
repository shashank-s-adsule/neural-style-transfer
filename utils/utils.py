import sys,os
import numpy as np
import cv2
import PIL
import torch, torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def preprocess(path,resize=None,device="cpu"):
    if not os.path.exists(path):
       raise Exception(f"image path does not exists: \u001b[1;33m{path}\u001b[0m")

    img=cv2.imread(path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)      #convert BGR->RGB  [can also use img[:,:,::-1]]

    # resize section
    if resize is not None:  
        if isinstance(resize,int):              # if only one dimesion is specifyed 
            H,W,C=img.shape
            new_H=resize
            new_W=int(W*(new_H/H))
            img=cv2.resize(img,(new_W,new_H),interpolation=cv2.INTER_CUBIC)
        else:      # if both dimesion are specified 
            img=cv2.resize(img,(resize[0],resize[1]),interpolation=cv2.INTER_CUBIC)

    img=img.astype(np.float32)
    img/=255.0

    transform=transforms.Compose([                          #HWC->NCHW
        transforms.ToTensor(),
        transforms.Lambda(lambda x:x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255,std=IMAGENET_STD_NEUTRAL)
    ])

    preprocess_img=transform(img).to(device).unsqueeze(0)
    return preprocess_img


def save_image(img,img_path):
    if(img.shape==2):
        img=np.stack((img)*3,axis=-1)
    cv2.imwrite(img_path,img)

def gernate_out_image_name(args):
    




if __name__=="__main__":
    # for debuging
    img_path=r"c:\Users\Shashank\Downloads\grayscale-image.jpg"
    pre_img=preprocess(img_path,resize=640)    
    print(pre_img.shape)

