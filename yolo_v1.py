import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import ctypes
import os
from torchvision import models
from PIL import Image
from pprint import pprint
from collections import OrderedDict

class YOLOv1(nn.Module):
    def __init__(self, class_names, grid_size, img_size=(448,448)):
        super(YOLOv1,self).__init__()
        self.num_bbox = 2
        self.input_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names.keys())        
        self.grid = grid_size

        # resnet50 = models.resnet50(pretrained=True)
        # for parameter in resnet50.parameters():
        #     parameter.requires_grad = False
        # self.feature_extractors = nn.Sequential(*list(resnet50.children())[:-2])

        self.feature_extractor = self.backbone("_SQUEEZENET_")

        self.final_conv = nn.Sequential(
            #NB: To add more conv layers, the batch size must be increased from 8 to higher
            nn.Conv2d(512, 1024, 3, bias=False), #2048 is the number of output filters from the last resnet bottleneck
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool2d((7,7)) #nn.AdaptiveAvgPool2d((1,1))
        )

        """
        For input to the linear layers, formerly we assumed a fixed image size of 
        448x448. Based on the `final_conv`, i know that the activation sizes here
        would be 12x12x256.
        To reduce the dependency on the input image size, we now use a 
        Global Average Pooling operation. Therefore, we can always assume the 
        input to our 1st linear layer would be `256`, the output channels of `final_conv`
        """
        self.linear_layers = nn.Sequential(
            nn.Linear(50176, 12544, bias=False), #9216
            nn.BatchNorm1d(12544),
            nn.Dropout(p=0.5), 
            nn.LeakyReLU(0.1),

            nn.Linear(12544, 3136, bias=False),
            nn.BatchNorm1d(3136),
            nn.LeakyReLU(0.1),

            nn.Linear(3136, (self.grid*self.grid) * ((self.num_bbox*5) + self.num_classes), bias=False), #1470
            nn.BatchNorm1d(1470),
            nn.Sigmoid()
        )
        #nn.Linear(1024, self.grid*self.grid * ((self.num_bbox*5) + self.num_classes)),
        

    def forward(self, x):

        activations = self.feature_extractor(x)
        activations = self.final_conv(activations)

        flattened = torch.flatten(activations)
        flattened = flattened.view(x.size()[0],-1) #resize it so that it is flattened by batch
        
        linearOutput = self.linear_layers(flattened)

        detectionTensor = linearOutput.view(-1, self.grid, self.grid, ((self.num_bbox * 5) + self.num_classes))

        return detectionTensor

    """
    Weight Initialisation will help present the network from stalling, where
    it doesn't learn anything at all by prevent vanishing/exploding gradients
    """
    def init_weights(self):
        # def init(layer, a=0.1):
        #     nn.init.kaiming_uniform_(layer.weight, a)
        #     if layer.bias is not None:
        #         layer.bias.data.fill_(0.01)
        
        gain_leaky_relu = nn.init.calculate_gain('leaky_relu', 0.1)
        gain_sig = nn.init.calculate_gain('sigmoid')

        # for layer in self.final_conv:
        #     if isinstance(layer, nn.Conv2d):
        #         init(layer, gain_leaky_relu)

        # init(self.linear_layers[0], gain_leaky_relu)
        # init(self.linear_layers[4], gain_sig)
        # init(self.linear_layers[0], gain_sig)

    def backbone(self, name):
        def squeezenet_forward(x):
            x = squeezenet.features(x)
            # x = squeezenet.classifier(x)
            return x

        if name == "_RESNET_50_":
            resnet50 = models.resnet50(pretrained=True)
            return nn.Sequential(*list(resnet50.children())[:-2])
        elif name == "_SQUEEZENET_":
            squeezenet = models.squeezenet1_1(pretrained=True)
            delattr(squeezenet, 'classifier')
            squeezenet.forward = squeezenet_forward
            return squeezenet