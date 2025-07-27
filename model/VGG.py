import sys,os
import numpy as np
import cv2
import torch, torch.nn as nn
from torchvision import models,transforms

class VGG16(nn.Module):
    def __init__(self,required_grad=False,show_progress=False):
        super(VGG16,self).__init__()
        self.backbone=models.vgg16(pretrained=True,progress=show_progress).features         # pretrained backbone of VGG16
        self.layer_names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        self.content_feature_maps_index=1                                                     # feature map index for content images
        self.style_feature_maps_indices=list(range(4))                                          # feature map index for style images 

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

class VGG19(nn.Module):
    def __init__(self,use_relu=True,required_grad=False,show_progress=False):
        super(VGG19,self).__init__()
        self.backbone=models.vgg19(pretrained=True,progress=show_progress).features                  # pretrained backbone of VGG19
        if use_relu:
            self.layer_name=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1']
            self.offset=1
        else:
            self.layer_name=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv4_2', 'conv5_1']
            self.offset=0

        self.content_feature_maps_index=4                                                              # feature map index for content images
        self.style_feature_maps_indices=list(range(6))                                                 # feature map index for style images

        # create slice node for extracting feature maps 
        self.slice_1=nn.Sequential()
        self.slice_2=nn.Sequential()
        self.slice_3=nn.Sequential()
        self.slice_4=nn.Sequential()
        self.slice_5=nn.Sequential()
        self.slice_6=nn.Sequential()

        # add pretrained layer from backbone
        for i in range(1+self.offset):
            self.slice_1.add_module(str(i),self.backbone[i])
        for i in range(1+self.offset,6+self.offset):
            self.slice_2.add_module(str(i),self.backbone[i])
        for i in range(6+self.offset,11+self.offset):
            self.slice_3.add_module(str(i),self.backbone[i])
        for i in range(11+self.offset,20+self.offset):
            self.slice_4.add_module(str(i),self.backbone[i])
        for i in range(20+self.offset,22):
            self.slice_5.add_module(str(i),self.backbone[i])
        for i in range(22,29+self.offset):
            self.slice_6.add_module(str(i),self.backbone[i])

        # freeze all layers
        if not required_grad:
            for p in self.parameters(): 
                p.requires_grad=False

    def forward(self,x):
        x=self.slice_1(x)
        layer1_1=x
        x=self.slice_2(x)
        layer2_1=x
        x=self.slice_3(x)
        layer3_1=x
        x=self.slice_4(x)
        layer4_1=x
        x=self.slice_5(x)
        conv4_2=x
        x=self.slice_6(x)
        layer5_1=x
        out_feature={"relu1_1":layer1_1,"relu2_1":layer2_1,"relu3_1":layer3_1,"relu4_1":layer4_1,"conv4_2":conv4_2,"relu5_1":layer5_1}
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