import sys,os, argparse
import numpy as np
import cv2

import torch, torch.nn as nn
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import torchvision

from model import * 
from utils import preprocess, prepare_model, gram_matrix, total_variation

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

    def build_loss(self,model, optimizing_img, target_representations, content_feature_index_name, style_feature_maps_indices_names):
        target_content_representation = target_representations[0]
        target_style_representation = target_representations[1]

        current_set_of_feature_maps = model(optimizing_img)

        # print(current_set_of_feature_maps.keys())

        current_content_representation = current_set_of_feature_maps[content_feature_index_name].squeeze(axis=0)
        content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

        # print(current_content_representation.shape)

        style_loss=0.0
        current_style_representation=[gram_matrix(current_set_of_feature_maps[layer_name]) for  layer_name in current_set_of_feature_maps if layer_name in style_feature_maps_indices_names]

        # print(len(current_style_representation))

        for gram_gt,gram_hat in zip(target_style_representation,current_style_representation):
            style_loss+=torch.nn.MSELoss(reduction='sum')(gram_gt[0],gram_hat[0])
        style_loss/=len(target_style_representation)

        # print(style_loss) 
        tv_loss = total_variation(optimizing_img)

        total_loss=args.content_weight*content_loss+args.style_weight*style_loss+args.tv_weight*tv_loss
        return total_loss,content_loss,style_loss,tv_loss

    def make_turning_step(self,model,optmizer,target_representations, content_feature_index_name, style_feature_maps_indices_name):
        def tuning_step(optimizing_img):
            total_loss,content_loss,style_loss,tv_loss=self.build_loss(model,optimizing_img,target_representations,content_feature_index_name,style_feature_maps_indices_name)
            # computer gradiants
            tv_loss.backward()
            # update parmeters and zeros gradients
            optmizer.step()
            optmizer.zero_grad()
            return total_loss,content_loss,style_loss,tv_loss
        return tuning_step


    def nst_image(self,model_name):
        # content and style image load
        cimg=preprocess(self.content_img_path,args.height,self.DEVICE)
        simg=preprocess(self.style_img_path,args.height,self.DEVICE)

        # Model load
        model=Load_model(model_name)()
        model=model.to(self.DEVICE)

        # create combination image 
        if args.init_method=="random":
            gaussian_noise_img=np.random.normal(loc=0,scale=90,size=cimg.shape).astype(np.float32)
            init_img=torch.from_numpy(gaussian_noise_img).float().to(self.DEVICE)
        elif args.init_method=="content":
            init_img=cimg
        else:
            style_img_resize=preprocess(self.style_img_path,np.asarray(cimg.shape[2:]),self.DEVICE)
            init_img=style_img_resize

        # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
        optimizing_img = Variable(init_img, requires_grad=True)

        # prepare model and layer indices
        model,content_feature_maps_index_name, style_feature_maps_indices_name= prepare_model(model)
        print(f"\u001b[1;35mUsing \u001b[33m{args.model}\u001b[1;35m in the optimization procedure. \u001b[0m")

        # print(content_feature_maps_index_name)
        # print(style_feature_maps_indices_name)

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
        target_representations=[target_content_representation,target_style_representation]

        # print(target_content_representation.shape)
        # for x in target_style_representation:
        #     print(x.shape)

        num_iter={"adam":3000,"lbfgs":1000}
        
        # ## optimizatino process start
        if args.optimizer=="adam":
            optimizer=Adam((optimizing_img,),lr=1e1)
            turning_step=self.make_turning_step(model,optimizer,target_representations,content_feature_maps_index_name[1],style_feature_maps_indices_name[1])
            for i in range(num_iter[args.optimizer]):
                total_loss,content_loss,style_loss,tv_loss=turning_step(optimizing_img)
                with torch.no_grad():
                    print(f"\u001b[1;33mAdam\u001b[0m | \u001b[1;34miteration: \u001b[0m{i:03} \u001b[1;31mTotal loss: \u001b[0m{total_loss.item():12.4f} \u001b[1;31mContent loss: \u001b[0m{args.content_weight*content_loss.item():12.4f} \u001b[1;31mStyle loss: \u001b[0m{args.style_weight*style_loss.item():12.4f} \u001b[1;31mContent loss: \u001b[0m{args.tv_weight*tv_loss.item():12.4f}")
                    

    def TEST(self):
        
        pass


def argument():

    arg=argparse.ArgumentParser()
    arg.add_argument("--c_image",type=str,default=".\\data\\content\\TamilContentImages\\C_image2.jpg",help="path for content image")
    arg.add_argument("--s_image",type=str,default=".\\data\\styles\\Artworks\\boat_sail_abstract.jpg",help="path for style image")
    arg.add_argument("--height",type=int,default=400,help="height of the content and style image")

    arg.add_argument("--content_weight", type=float, help="weight factor for content loss", default=1e5)
    arg.add_argument("--style_weight", type=float, help="weight factor for style loss", default=3e4)
    arg.add_argument("--tv_weight", type=float, help="weight factor for total variation loss", default=1e0)

    arg.add_argument("--optimizer",type=str,choices=["lbfgs","adam"],default="adam")
    # arg.add_argument("--optimizer",type=str,choices=["lbfgs","adam"],default="lbfgs")
    arg.add_argument("--model",type=str,default="vgg16",help="default model for NST")
    arg.add_argument("--init_method",type=str,choices=["random","content","style"],default="content",help="...")
    arg.add_argument("--saving_freq", type=int, help="saving frequency for intermediate images (-1 means only final)", default=-1)

    return arg.parse_args()

if __name__=="__main__":
    args=argument()

    if "temp" not in os.listdir(): os.makedirs("temp")

    obj=NST(args)
    # obj.metadata()
    
    obj.nst_image("vgg16")  

    # obj.TEST()

    