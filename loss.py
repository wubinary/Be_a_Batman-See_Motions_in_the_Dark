from collections import namedtuple
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg


################################################################
##################  Perceptron loss Function  ##################

LossOutput = namedtuple("LossOutput", ["relu4", "relu5"])

class VggFeatures(nn.Module):
    def __init__(self):
        super(VggFeatures, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '26': "relu4",
            '35': "relu5",
        }
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)
    
class Perceptual_Loss(nn.Module):
    def __init__(self):
        super(Perceptual_Loss, self).__init__()
        self.extract_vgg_features = VggFeatures()
        self.mse_loss = nn.MSELoss()
        self.trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    def forward(self, predict, target):
        # The images have to be loaded in to a range of [0, 1]
        # predict: (B, D, C, H, W)
        # target: (B, C, H, W)
        if len(predict.shape)==5:
            b, d, c, h, w = predict.shape
            predict = predict.view(b*d, c, h, w)
            _, _, h, w = target.shape
            target = target.unsqueeze(1).expand(b, d, c, h, w)
            target = target.contiguous().view(b*d, c, h, w)
        _, _, h, w = predict.shape
        predict = self.extract_vgg_features(F.interpolate(predict, size=(h//2, w//2), mode='bicubic'))
        target = self.extract_vgg_features(F.interpolate(target, size=(h//2, w//2), mode='bicubic'))
        losses = []
        for i in range(2):
            losses.append(self.mse_loss(predict[i],target[i]))
        torch.cuda.empty_cache()
        return sum(losses)

class Consistency_Loss(nn.Module):
    def __init__(self):
        super(Consistency_Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        #self.l1_loss = nn.L1Loss()
        pass 

    def forward(self, predict, target):
        # The pixel value of image have to be in the range of [0,1]
        # predict: (B, D, C, H, W)
        # target: (B, C, H, W)
        if len(predict.shape)==5:
            b, d, c, h, w = predict.shape
            predict = predict.view(b*d, c, h, w)
            _, _, h, w = target.shape
            target = target.unsqueeze(1).expand(b, d, c, h, w)
            target = target.contiguous().view(b*d, c, h, w)
        _, _, h, w = predict.shape 
        predict = F.interpolate(predict, size=(h,w), align_corners=True, 
                                                    mode='bilinear') #'bicubic')
        target = F.interpolate(target, size=(h,w), align_corners=True, 
                                                    mode='bilinear') #'bicubic')
        loss = self.mse_loss(predict, target) 
        #loss = self.l1_loss(predict, target)
        return loss  
