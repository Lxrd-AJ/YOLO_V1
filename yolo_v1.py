import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import ctypes
import os
from torchvision import models
from PIL import Image
# from utilities import convert_center_coords_to_noorm, max_box, convert_cls_idx_name, confidence_threshold #parse_config
from pprint import pprint
from collections import OrderedDict


class Yolo_V1(nn.Module):
    def __init__(self, class_names, grid_size, img_size=(448,448)):
        super(Yolo_V1,self).__init__()
        self.num_bbox = 2
        self.input_size = img_size
        self.class_names = class_names
        self.num_classes = len(class_names.keys())        
        self.grid = grid_size
        # self.blocks = blocks
        # self.extraction_layers, extract_out = self.parse_conv(blocks)
        resnet50 = models.resnet50(pretrained=True)
        self.extraction_layers = nn.Sequential(*list(resnet50.children())[:-2])
        

        # self.final_conv = nn.Conv2d(extract_out, 256, 3, 1, 1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(2048, 2048, 3, 1, 1), #2048 is the number of output filters from the last resnet bottleneck
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(2048, 1024, 3, 1, 1), 
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, 1, 1), 
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 512, 3, 1, 0),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.AdaptiveAvgPool2d((1,1))
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
            nn.Linear(512,1024,True),
            nn.Dropout(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, self.grid*self.grid * ((self.num_bbox*5) + self.num_classes)),
            nn.Sigmoid()
        )
        

    def forward(self, x):

        actv = self.extraction_layers(x)
        actv = self.final_conv(actv)                
        
        # actv = x
        # for i in range(len(self.extraction_layers)):
        #     actv = self.extraction_layers[i](actv)                        
        #     assert not torch.isnan(actv).any()        

        lin_inp = torch.flatten(actv)
        lin_inp = lin_inp.view(x.size()[0],-1) #resize it so that it is flattened by batch        
        lin_out = self.linear_layers(lin_inp)
        # lin_out = torch.sigmoid(lin_out) #TODO: Move this to where the layers are constructed         
        det_tensor = lin_out.view(-1,((self.num_bbox * 5) + self.num_classes),self.grid,self.grid)
        return det_tensor

    """
    Weight Initialisation will help present the network from stalling, where
    it doesn't learn anything at all
    """
    def init_weights(self):
        def init(layer, a=0.1):
            nn.init.kaiming_uniform_(layer.weight, a)
            layer.bias.data.fill_(0.01)
        
        gain_leaky_relu = nn.init.calculate_gain('leaky_relu', 0.1)
        gain_sig = nn.init.calculate_gain('sigmoid')

        for layer in self.final_conv:
            if isinstance(layer, nn.Conv2d):
                init(layer, gain_leaky_relu)

        init(self.linear_layers[0], gain_leaky_relu)
        init(self.linear_layers[3], gain_sig)        



    """
    Returns the convolutional blocks in the YOLO module architecture
    This is based on the extraction net architecture as described here https://pjreddie.com/darknet/imagenet/#extraction
    """
    # def seq_conv(self, item, module, idx, in_channels):
    #     padding = int(item['pad'])
    #     kernel_size = int(item['size'])
    #     stride = int(item['stride'])               
    #     pad = int(item['pad'])
    #     filters = int(item['filters'])
    #     activation = item['activation']
    #     try:
    #         batch_norm = bool(item['batch_normalize'])
    #     except:
    #         batch_norm = False
    #     bias = False if batch_norm else True

    #     module.add_module(f'conv_{idx}', nn.Conv2d(in_channels, filters, kernel_size, stride, pad, bias=bias))
    #     if batch_norm:
    #         module.add_module(f"batch_norm_{idx}", nn.BatchNorm2d(filters))

    #     if activation == 'leaky':
    #         module.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1, inplace=True))
    #     else:
    #         print("Unknown activation function provided for YOLO v1")

    #     return module, filters

    def parse_conv(self, blocks):
        conv_layers = nn.ModuleList()              
        prev_filters = 3 #image of 3 channels        
        for idx, item in enumerate(blocks[1:-1]):
            module = nn.Sequential()
            
            if item['type'] == 'convolutional': 
                module, prev_filters = self.seq_conv(item, module, idx, prev_filters)
            elif item['type']  == 'maxpool':
                maxpool = nn.MaxPool2d(int(item['size']), int(item['stride']))
                module.add_module(f"maxpool_{idx}", maxpool)
            elif item['type'] == 'avgpool':
                module.add_module(f"avgpool_{idx}", nn.AvgPool2d(2,2))

            conv_layers.append(module)
        return conv_layers, prev_filters

    """    
    """
    def load_batchnorm_weights(self, weights, batch_norm, amount_read):         
        amount_bn_bias = batch_norm.bias.numel()                 
        
        assert amount_bn_bias == batch_norm.weight.numel()
        assert amount_bn_bias == batch_norm.running_mean.numel()
        assert amount_bn_bias == batch_norm.running_var.numel()
                                
        bn_bias = torch.from_numpy(weights[amount_read:amount_read+amount_bn_bias])      
        amount_read += amount_bn_bias
                
        bn_weights = torch.from_numpy(weights[amount_read:amount_read+amount_bn_bias])
        amount_read += amount_bn_bias
        
        bn_run_mean = torch.from_numpy(weights[amount_read:amount_read+amount_bn_bias])
        amount_read += amount_bn_bias
        
        bn_run_var = torch.from_numpy(weights[amount_read:amount_read+amount_bn_bias])
        amount_read += amount_bn_bias                
        bn_bias = bn_bias.view_as(batch_norm.bias.data)
        bn_weights = bn_weights.view_as(batch_norm.weight.data)
        bn_run_mean = bn_run_mean.view_as(batch_norm.running_mean)
        bn_run_var = bn_run_var.view_as(batch_norm.running_var)

        # copy the loaded params into the bias params
        batch_norm.bias.data.copy_(bn_bias)
        batch_norm.weight.data.copy_(bn_weights)
        batch_norm.running_mean.copy_(bn_run_mean)
        batch_norm.running_var.copy_(bn_run_var)
        
        return batch_norm, amount_read

    def load_conv_weights(self, weights, conv, idx=0):
        # copy the bias if it conv layer has bias
        if conv.bias is not None:
            num_conv_bias = conv.bias.numel()                    
            conv_bias = torch.from_numpy(weights[idx:idx+num_conv_bias])
            idx += num_conv_bias
            conv_bias = conv_bias.view_as(conv.bias.data)
            conv.bias.data.copy_(conv_bias)
        # copy the conv weights
        num_conv_weights = conv.weight.numel()                
        conv_weights = torch.from_numpy(weights[idx:idx+num_conv_weights])
        idx += num_conv_weights
        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)
        return conv, idx   
            

    def load_extraction_weights(self, weights_file):        
        with open(weights_file,'rb') as file:
            #NB: An internal file pointer is maintained, so read the header first
            major, minor, revision = np.fromfile(file, dtype=np.int32, offset=0, count=3)    
            if ((major*10 + minor) >= 2 and major < 1000 and minor < 1000):
                idx = 12 + ctypes.sizeof(ctypes.c_size_t)                
            else:                
                idx = 16            

            weights = np.fromfile(file, dtype=np.float32, offset=idx, count=-1)            
            # num_elements = sum(p.numel() for p in self.extraction_layers.parameters() if p.requires_grad)
            # print(f"Num Elements in extraction layers: {num_elements}")            
            # populate the convolutional layers                                   
            for i in range(len(self.extraction_layers)):
                conv_block = self.extraction_layers[i]                                
                if len(conv_block) == 3: #common block with conv, batch norm and leaky relu
                    conv_block[1], idx = self.load_batchnorm_weights(weights, conv_block[1], idx)
                    conv_block[0], idx = self.load_conv_weights(weights, conv_block[0], idx)      
                elif len(conv_block) == 2: #block with conv and leaky relu
                    conv_block[0], idx = self.load_conv_weights(weights, conv_block[0], idx)
                else:
                    continue            