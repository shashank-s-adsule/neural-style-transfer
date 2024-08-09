import sys,os, argparse
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision

from model.VGG import VGG16
from utils import preprocess

def argument():

    arg=argparse.ArgumentParser()
    arg.add_argument("--c_image",type=str,default="D:\\shashank\\test code\\test_repo\\neural-style-transfer\\data\\dataset\\content\\TamilContentImages\\C_image2.jpg",help="path for content image")
    arg.add_argument("--s_image",type=str,default="D:\\shashank\\test code\\test_repo\\neural-style-transfer\\data\\dataset\\styles\\Artworks\\423786.jpg",help="path for style image")
    arg.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu", help="set device cofigation (cuda, CPU)")

    return arg.parse_args()

if __name__=="__main__":
    args=argument()
    print(args.c_image)