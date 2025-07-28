import sys,os, argparse
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision

from model import * 
from utils import preprocess, prepare_model, gram_matrix

class NST:
    def __init__(self,args):
        self.content_img_path=args.c_image
        self.style_img_path=args.s_image
        self.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
        # temp image
        if "img" not in os.listdir("./temp"): os.makedirs("./temp/img")
        self.out_img_name=f"combine_{os.path.splitext(os.path.basename(self.content_img_path))[0]}_AND_{os.path.basename(self.style_img_path)}"

    def metadata(self):
        
        print(f"\u001b[1;35m{'Metadata about input images':-^60}\u001b[0m")
        # metadata of content image 
        print("\u001b[1;33mContent Image:\u001b[0m")
        cimg=cv2.imread(self.content_img_path)
        print(f"\u001b[1;32mimage path:\u001b[0m{self.content_img_path}")
        print(f"\u001b[1;32mimage dimmesion: \u001b[0m{cimg.shape[0]} X {cimg.shape[1]}")
        
        # metadate of style image
        print("\n\u001b[1;33mStyle Image:\u001b[0m")
        simg=cv2.imread(self.style_img_path)
        print(f"\u001b[1;32mimage path:\u001b[0m{self.style_img_path}")
        print(f"\u001b[1;32mimage dimmesion: \u001b[0m{simg.shape[0]} X {simg.shape[1]}")
        
        print(f"\u001b[1;35m{'Metadata about Model':-^60}\u001b[0m")
        print(f"\u001b[1;33mModel:\u001b[0m\t{args.model}")
        print(f"\u001b[1;33mDevice:\u001b[0m\t{self.DEVICE}")


    def nst_image(self,model_name):
        # content and style image load
        cimg=preprocess(self.content_img_path,None,self.DEVICE)
        simg=preprocess(self.style_img_path,None,self.DEVICE)

        # Model load
        model=Load_model(model_name)()
        model=model.to(self.DEVICE)
        # prepare model and layer indices
        model,content_feature_maps_index_name, style_feature_maps_indices_name= prepare_model(model)
        print(f"\u001b[1;35mUsing \u001b[33m{args.model}\u001b[1;35m in the optimization procedure. \u001b[0m")

        print(content_feature_maps_index_name)
        print(style_feature_maps_indices_name)

        # extract feature maps of the images            (dict() so change the following assigment acordinly)
        content_img_set_of_feature_maps=model(cimg)
        style_img_set_of_feature_maps=model(simg)
        
        # for k,v in content_img_set_of_feature_maps.items():
        #     print(f"\u001b[33m{k}:\u001b[0m {v.shape}")
        # print()
        # for k,v in style_img_set_of_feature_maps.items():
        #     print(f"\u001b[33m{k}:\u001b[0m {v.shape}")
        # print()
        

        target_content_representation= content_img_set_of_feature_maps[content_feature_maps_index_name[1]].squeeze(axis=0)
        target_style_representation=[gram_matrix(style_img_set_of_feature_maps[layer_name]) for index, layer_name in enumerate(style_img_set_of_feature_maps) if layer_name in style_feature_maps_indices_name[1]]
        
        # for x in target_style_representation:
        #     print(x.shape)

    def TEST(self):
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
    
    obj.nst_image("vgg16")

    # obj.TEST()

    