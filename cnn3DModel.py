import torch
import numpy as np
import torch.nn as nn

class conv3DCNN(nn.Module):
    def __init__(self):
        super(conv3DCNN, self).__init__()
        self.conv_layer_rgb1 = self.make_conv_layer1(3, 16)
        self.conv_layer_rgb2 = self.make_conv_layer1(16, 16)
        self.conv_layer_rgb3 = self.make_conv_layer1(16, 32)
        self.conv_layer_rgb4 = self.make_conv_layer1(32, 32)

        self.conv_layer_op_flow1 = self.make_conv_layer1(1, 16)
        self.conv_layer_op_flow2= self.make_conv_layer1(16, 16)
        self.conv_layer_op_flow3 = self.make_conv_layer1(16, 32)
        self.conv_layer_op_flow4 = self.make_conv_layer1(32, 32)

        self.fusion_maxPool = nn.MaxPool3d((8, 1, 1))



    def make_conv_layer1(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3)),
            nn.LeakyReLU(),
            nn.Conv3d(out_c, out_c, kernel_size=(3, 1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        return conv_layer

    def forward(self, rgb, opFlow):
        rgb = self.conv_layer_rgb1(rgb)
        rgb = self.conv_layer_rgb2(rgb)
        rgb = self.conv_layer_rgb3(rgb)
        rgb = self.conv_layer_rgb4(rgb)
        rgb = nn.LeakyReLU(rgb)

        opFlow = self.conv_layer_op_flow1(opFlow)
        opFlow = self.conv_layer_op_flow2(opFlow)
        opFlow = self.conv_layer_op_flow2(opFlow)
        opFlow = self.conv_layer_op_flow4(opFlow)
        opFlow = nn.Sigmoid(opFlow)

        fusion = rgb * opFlow
        fusion = self.fusion_maxPool(fusion)


