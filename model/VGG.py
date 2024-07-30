import sys,os
import numpy as np
import cv2
import torch, torch.nn as nn

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        self.backbone=nn.Sequential(
            nn.Conv2d(3,8,3,1,1),
            nn.ReLU(True)
        )

    def forward(self,x):
        out1=self.backbone(x)
        return out1

if __name__=="__main__":
    model=VGG()
    data=torch.randn(1,3,640,640)
    pred=model(data)
    print(f"data: {pred.shape}")