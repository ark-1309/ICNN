import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self, DEVICE='cuda', layer_names=['3', '8', '15', '22']):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features.to(DEVICE)
        self.layer_names = layer_names
        self.eval()
        
    def __call__(self, in_1, in_2):
        assert in_1.shape == in_2.shape
        loss = 0.
        out_1 = in_1.to('cpu'); out_2 = in_2.to('cpu')
        for name, module in self.vgg_layers._modules.items():
            out_1 = module(out_1); out_2 = module(out_2);
            if name in self.layer_names:
                loss += F.mse_loss(out_1, out_2, reduction='none').flatten(start_dim=1).mean(dim=1)
        return loss[0]