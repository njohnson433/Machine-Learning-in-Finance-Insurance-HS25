import torch
import torch.nn as nn
import torchvision



# Function borrowed from
# https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/vision/model_getter.py
def get_resnet(model_name, weights=None, **kwargs):
    """
    size: 18, 34, 50
    """
    func = getattr(torchvision.models, model_name)
    resnet = func(weights=weights, **kwargs)
    resnet.encoding_dim = resnet.fc.in_features
    resnet.fc = torch.nn.Identity()
    return resnet



if __name__ == "__main__":
    model = get_resnet("resnet18")
    x = torch.randn(2, 3, 224, 224)
    