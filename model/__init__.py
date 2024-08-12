from .VGG import VGG19, VGG16
__all__=["prepare_model"]

model_list={"vgg16":VGG16,"vgg19":VGG19}

def prepare_model(model_name):
    model_name=model_name.lower()
    if(model_name in model_list.keys()): return model_list[model_name]
    else:
        raise Exception(f"model \u001b[1;31m{model_name}\u001b[0m  dosen't exist in the models directory")