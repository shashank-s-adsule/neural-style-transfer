import sys,os
import numpy as np
import cv2
import torch, torch.nn as nn
from torchvision import models,transforms

class VGG16(nn.Module):
    def __init__(self,required_grad=False,show_progress=False):
        super(VGG16,self).__init__()
        self.backbone=models.vgg16(pretrained=True,progress=show_progress).features         # pretrained backbone of VGG16
        self.content_feature_maps_idx=1                                                     # feature map index for content images
        self.style_feature_maps_idx=list(range(4))                                          # feature map index for style images 

        # create slice node for extracting feature maps 
        self.slice_1=nn.Sequential()
        self.slice_2=nn.Sequential()
        self.slice_3=nn.Sequential()
        self.slice_4=nn.Sequential()
        
        # add pretrained layer from backbone  
        for i in range(4):
            self.slice_1.add_module(str(i),self.backbone[i])
        for i in range(4,9):
            self.slice_2.add_module(str(i),self.backbone[i])
        for i in range(9,16):
            self.slice_3.add_module(str(i),self.backbone[i])
        for i in range(16,23):
            self.slice_4.add_module(str(i),self.backbone[i])

        # freeze all layers 
        if not required_grad:
            for p in self.parameters(): 
                p.requires_grad=False

    def forward(self,x):
        x=self.slice_1(x)
        relu1_2=x
        x=self.slice_2(x)
        relu2_2=x
        x=self.slice_3(x)
        relu3_3=x
        x=self.slice_4(x)
        relu4_3=x
        out_feature={"relu1_2":relu1_2,"relu2_2":relu2_2,"relu3_3":relu3_3,"relu4_3":relu4_3}
        return out_feature

# remaining 
class VGG19(nn.Module):
    def __init__(self,required_grad=False,show_progress=False):
        super(VGG19,self).__init__()
        self.backbone=models.vgg19(pretrained=True,progress=show_progress).features                  # pretrained backbone of VGG19
        self.content_feature_maps_idx=1                                                              # feature map index for content images
        self.style_feature_maps_idx=list(range(5))                                                   # feature map index for style images

        # create slice node for extracting feature maps 
        self.slice_1=nn.Sequential()
        self.slice_2=nn.Sequential()
        self.slice_3=nn.Sequential()
        self.slice_4=nn.Sequential()
        self.slice_5=nn.Sequential()

        # add pretrained layer from backbone
        for i in range(4):
            self.slice_1.add_module(str(i),self.backbone[i])
        for i in range(4,9):
            self.slice_2.add_module(str(i),self.backbone[i])
        for i in range(9,18):
            self.slice_3.add_module(str(i),self.backbone[i])
        for i in range(18,27):
            self.slice_4.add_module(str(i),self.backbone[i])
        for i in range(27,36):
            self.slice_5.add_module(str(i),self.backbone[i])

        # freeze all layers
        if not required_grad:
            for p in self.parameters(): 
                p.requires_grad=False

    def forward(self,x):
        x=self.slice_1(x)
        relu1_2=x
        x=self.slice_2(x)
        relu2_2=x
        x=self.slice_3(x)
        relu3_4=x
        x=self.slice_4(x)
        relu4_4=x
        x=self.slice_5(x)
        relu5_4=x
        out_feature={"relu1_2":relu1_2,"relu2_2":relu2_2,"relu3_4":relu3_4,"relu4_4":relu4_4,"relu5_4":relu5_4}
        return out_feature


if __name__=="__main__":
    os.system("cls" if os.name=="nt" else "clear")
    # for checking and debuging
    img=torch.randn(1,3,640,640)
    while True:
        n=int(input("enter the choices\n1. for checking VGG16\n2. for checking VGG19\n3. exit\n"))
        os.system("cls" if os.name=="nt" else "clear")
        match n:
            case 1:
                m=VGG16()       
                pred=m(img)
            case 2:
                m=VGG19()
                pred=m(img)
            case 3:
                exit(1)
            case _:
                print("\u001b[1;34m[\u001b[1;31mERROR\u001b[1;34m]\u001b[1;31m:\u001b[0m invalid input")
                exit(1)
        
        # for checking image shape after each slice-convolution  
        for key,val in pred.items():
            print(f" \u001b[1;33m{key}\u001b[0m: {val.shape}")

        # for going back to previous step 
        exit_code=input("press 'back' or '<' to go previous step: ")
        if exit_code not in ["back","<"]: break
        os.system("cls" if os.name=="nt" else "clear")