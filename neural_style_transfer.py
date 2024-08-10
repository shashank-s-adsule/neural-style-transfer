import sys,os, argparse
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision

from model.VGG import VGG16
# from utils import preprocess

class NST:
    def __init__(self,args):
        self.content_img_path=args.c_image
        self.style_img_path=args.s_image
        self.device="cuda" if torch.cuda.is_available() else "cpu"
    
    def metadata(self):
        # metadata of content image 
        print(f"{'Metadata about input images':-^60}")
        print("\u001b[1;33mContent Image:\u001b[0m")
        print(f"\u001b[1;32mimage path:\u001b[0m {self.content_img_path}")
        c_img=cv2.imread(self.content_img_path)
        print(f"\u001b[1;33mimage dimmesion: {c_img.shape[0]}X{c_img.shape[1]}\u001b[0m")
        # metadate of style image
        print("\n\u001b[1;33mStyle Image:\u001b[0m")
        print(f"\u001b[1;32mimage path:\u001b[0m {self.style_img_path}")
        s_img=cv2.imread(self.style_img_path)
        print(f"\u001b[1;33mimage dimmesion: {s_img.shape[0]}X{s_img.shape[1]}\u001b[0m")
        pass


def argument():

    arg=argparse.ArgumentParser()
    arg.add_argument("--c_image",type=str,default="D:\\shashank\\test code\\test_repo\\neural-style-transfer\\data\\dataset\\content\\TamilContentImages\\C_image2.jpg",help="path for content image")
    arg.add_argument("--s_image",type=str,default="D:\\shashank\\test code\\test_repo\\neural-style-transfer\\data\\styles\\Artworks\\boat_sail_abstract.jpg",help="path for style image")

    return arg.parse_args()

if __name__=="__main__":
    args=argument()

    obj=NST(args)
    obj.metadata()
    