import sys,os, argparse
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision

from model import * 
from utils import preprocess, prepare_model

class NST:
    def __init__(self,args):
        self.content_img_path=args.c_image
        self.style_img_path=args.s_image
        self.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        # temp image
        if "img" not in os.listdir("./temp"): os.makedirs("./temp/img")
        self.out_img_name=f"combine_{os.path.splitext(os.path.basename(self.content_img_path))[0]}_AND_{os.path.basename(self.style_img_path)}"

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
        # content and style image load
        cimg=preprocess(self.content_img_path,None,self.DEVICE)
        simg=preprocess(self.style_img_path,None,self.DEVICE)

        # Model load
        model=Load_model("vgg16")()
        model=model.to(self.DEVICE)

        model,content_feature_maps_index_name, style_feature_maps_indices_name= prepare_model(model)
        

        # print(model)

    def TEST(self):
        model=Load_model("vgg16")()
        model=model.to(self.DEVICE)

        a,b,c=prepare_model(model)

        print(b,c)

        pass


def argument():

    arg=argparse.ArgumentParser()
    arg.add_argument("--c_image",type=str,default=".\\data\\content\\TamilContentImages\\C_image2.jpg",help="path for content image")
    arg.add_argument("--s_image",type=str,default=".\\data\\styles\\Artworks\\boat_sail_abstract.jpg",help="path for style image")
    arg.add_argument("--model",type=str,default="vgg16",help="default model for NST")

    return arg.parse_args()

if __name__=="__main__":
    args=argument()

    if "temp" not in os.listdir(): os.makedirs("temp")

    obj=NST(args)
    # obj.metadata()
    
    # obj.nst_image("anc")

    obj.TEST()

    