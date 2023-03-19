# code adopted https://github.com/xuchen-ethz/neural_object_fitting

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class PerceptualLoss(nn.Module):
    def __init__(self, type='l2', reduction='mean', final_layer=14):
        super(PerceptualLoss, self).__init__()
        self.model = self.contentFunc(final_layer=final_layer)
        self.model.eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.type = type
        if type == 'l1':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
        elif type == 'l2':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        elif type == 'both':
            self.criterion1 = torch.nn.L1Loss(reduction=reduction)
            self.criterion2 = torch.nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError

    @staticmethod
    def contentFunc(final_layer=14):
        cnn = torchvision.models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == final_layer:
                break
        return model

    def __call__(self, fakeIm, realIm):
        f_fake = self.model.forward(self.normalize(fakeIm))
        f_real = self.model.forward(self.normalize(realIm))
        if self.type == 'both':
            loss = self.criterion1(f_fake, f_real.detach()) + self.criterion2(f_fake, f_real.detach())
        else:
            loss = self.criterion(f_fake, f_real.detach())
        return loss
