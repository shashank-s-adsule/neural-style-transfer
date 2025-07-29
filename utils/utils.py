import sys,os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch, torch.nn as nn
from torchvision import transforms

'''----------------------------------------| Image utils function |----------------------------------------'''
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def preprocess(path,resize=None,device="cpu"):
    if not os.path.exists(path):
       raise Exception(f"image path does not exists: \u001b[1;33m{path}\u001b[0m")

    img=cv2.imread(path)
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)      #convert BGR->RGB  [can also use img[:,:,::-1]]
    
    ## part 1 
    # resize section
    if resize is not None:  
        if isinstance(resize,int) and resize!=-1:              # if only one dimesion is specifyed 
            H,W,C=img.shape
            new_H=resize
            new_W=int(W*(new_H/H))
            img=cv2.resize(img,(new_W,new_H),interpolation=cv2.INTER_CUBIC)
        else:      # if both dimesion are specified 
            img=cv2.resize(img,(resize[0],resize[1]),interpolation=cv2.INTER_CUBIC)

    img=img.astype(np.float32)
    img/=255.0

    ## part2
    ## normalize using ImageNet's mean
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
    pass

def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')

def save_maybe_display(optimizing_img, dump_path, saving_freq, img_id, num_of_iterations,out_image_name=None, should_display=False,):
    out_img=optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img=np.moveaxis(out_img,0,2)
    if img_id==num_of_iterations-1 or (saving_freq>0 and img_id%saving_freq==0):
        out_img_name=f"{str(img_id).zfill(4)}.jpg" if saving_freq!=-1 else out_image_name
        dump_img=np.copy(out_img)
        dump_img+=np.array(IMAGENET_MEAN_255).reshape((1,1,3))
        dump_img=np.clip(dump_img,0,255).astype('uint8')
        cv2.imwrite(os.path.join(dump_path,out_img_name),dump_img[:,:,::-1])
    
    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()

'''----------------------------------------| Model utils functions |----------------------------------------'''
def prepare_model(model):
    content_feature_map_index=model.content_feature_maps_index
    style_feature_map_indices=model.style_feature_maps_indices
    layer_names=model.layer_names

    content_fms_index_name = (content_feature_map_index, layer_names[content_feature_map_index])
    style_fms_indices_names = (style_feature_map_indices, layer_names)  

    return model,content_fms_index_name,style_fms_indices_names

def gram_matrix(x,shoudl_normalize=True):
    (B,C,H,W)=x.size()
    feature=x.view(B,C,W*H)
    feature_t=feature.transpose(1,2)
    gram=feature.bmm(feature_t)
    if shoudl_normalize:
        gram/=(C*H*W)
    return gram

def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

if __name__=="__main__":
    # for debuging
    # img_path=r"c:\Users\Shashank\Downloads\grayscale-image.jpg"
    # pre_img=preprocess(img_path,resize=640)    
    # print(pre_img.shape)

    T=torch.randn([1,3,64,64])
    G=gram_matrix(T)
    print(G.shape)