import sys,os, argparse
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision

from model import * 
# from utils import preprocess

class NST:
    def __init__(self,args):
        self.content_img_path=args.c_image
        self.style_img_path=args.s_image
        self.device="cuda" if torch.cuda.is_available() else "cpu"
    
    def metadata(self):
        print(f"{'Metadata about input images':-^60}")
        
        # metadata of content image 
        print("\u001b[1;33mContent Image:\u001b[0m")
        c_img=cv2.imread(self.content_img_path)
        print(f"\u001b[1;32mimage path:\u001b[0m{self.content_img_path}")
        print(f"\u001b[1;32mimage dimmesion: \u001b[0m{c_img.shape[0]} X {c_img.shape[1]}")
        # metadate of style image

        print("\n\u001b[1;33mStyle Image:\u001b[0m")
        s_img=cv2.imread(self.style_img_path)
        print(f"\u001b[1;32mimage path:\u001b[0m{self.style_img_path}")
        print(f"\u001b[1;32mimage dimmesion: \u001b[0m{s_img.shape[0]} X {s_img.shape[1]}")
        
    def nst_image(self,model_name):
        model=prepare_model("vgg161")()

        print(model)



def argument():

    arg=argparse.ArgumentParser()
    arg.add_argument("--c_image",type=str,default=".\\data\\content\\TamilContentImages\\C_image2.jpg",help="path for content image")
    arg.add_argument("--s_image",type=str,default=".\\data\\styles\\Artworks\\boat_sail_abstract.jpg",help="path for style image")
    arg.add_argument("--model",type=str,default="vgg16",help="default model for NST")

    return arg.parse_args()

if __name__=="__main__":
    args=argument()

    obj=NST(args)
    # obj.metadata()
    obj.nst_image("anc")
    