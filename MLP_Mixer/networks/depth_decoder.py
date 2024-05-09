from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        
        #print(self.scales)
        #print(omer)
        for i in range(2, -1, -1):

            x = self.convs[("upconv", i, 0)](x)
            #print(x.shape)
            #print(upsample(x).shape)
            x = [upsample(x)]
            

            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
                #print(input_features[i - 1].shape)
                
            x = torch.cat(x, 1)
            #print(x.shape)
            x = self.convs[("upconv", i, 1)](x)
            #print(x.shape)

            if i in self.scales:
                #print("*******************")
            
                f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
                #print("f:",f.shape)
                
                self.outputs[("disp", i)] = self.sigmoid(f)
            #print(x.shape)
        #print(omer)
        
        return self.outputs

