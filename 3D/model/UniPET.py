##UniPET: a universal network for high-quality PET image denoising across varied dose reduction factors 
##https://github.com/Yaziwel/UniPET

import torch
from torch import nn
import torch.nn.functional as F 
import numpy as np
# from Blur import Blur3D

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.se = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
    )


    def forward(self, x):
        # b, c, _, _ ,_= x.size()
        y = self.avg_pool(x) 
        
        # import pdb 
        # pdb.set_trace()

        y = self.se(y)
        out = y*x

        return out



class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel_size, kernel_size, kernel_size)))
        self.EPS = 1e-8
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu') 
        
        # self.bias = nn.Parameter(torch.zeros((out_chan)))

    def _get_same_padding(self, size, kernel_size, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2

    def forward(self, x, sw):
        b, c, d, h, w = x.shape
        w1 = sw[:, None, :, None, None, None]
        w2 = self.weight[None, :, :, :, :,:]
        weights = w2 * (w1+1)
        de_w = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.EPS)
        weights = weights * de_w 
        
        # b1 = sb
        # b2 = self.bias.unsqueeze(0)
        # biases = b2*b1 
        # de_b = torch.rsqrt((biases ** 2).sum(dim=1, keepdim=True) + self.EPS) 
        # biases = biases*de_b 
        # biases = biases.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
        
        x = x.reshape(1, -1, d, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)
        padding = self._get_same_padding(d, self.kernel_size, self.dilation, self.stride)
        x = F.conv3d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.filters, d, h, w) 
        # x = x + biases
        # 
        
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out, 
        proj_factor=4,
        activation = nn.PReLU()
    ):
        super().__init__() 
        attn_dim_in = dim//proj_factor

        self.conv1 = nn.Sequential(
            nn.Conv3d(dim, attn_dim_in, 1, bias = False),
            activation
            ) 
        self.att = ChannelAttention(attn_dim_in)

        self.conv2=nn.Sequential(
            nn.Conv3d(attn_dim_in, attn_dim_in, 3, 2, 1, bias = False),
            activation,
            nn.Conv3d(attn_dim_in, dim_out, 1, bias = False),
            activation
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)), 
            nn.Conv3d(dim_out, 256, 1,bias=True), 
            activation
            )
        
    def forward(self, x): 
        
        out = self.conv1(x) 
        out = self.att(out)
        out = self.conv2(out)
        latent = self.fc(out)
        return out, latent

class SAN(nn.Module):
    def __init__(self,num_adapters=1):
        super(SAN, self).__init__()
        down = 4
        get_channel = [64, 256, 256, 256, 256]

        self.body=nn.ModuleList([EncoderBlock(get_channel[i],get_channel[i+1]) for i in range(down)])


    def forward(self, x): 
        
        latent_w_list = [] 
        
        # import pdb 
        # pdb.set_trace() 
        latent_list = []

        for _, encoder in enumerate(self.body):
            x, latent = encoder(x) 

            latent_list.append(latent)
            
        # x = self.avg_pool(x) 
        
        # latent_w = self.fc_after_mapping(x)


        return latent_list

class ResModBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, latent_size, activation=nn.PReLU()):
        super(ResModBlock, self).__init__()
        self.convmod1=Conv3DMod(n_feats, n_feats, kernel_size)
        self.act=activation
        self.convmod2=Conv3DMod(n_feats, n_feats, kernel_size)
        self.res_scale = nn.Parameter(torch.ones((1, n_feats, 1, 1, 1)), requires_grad=True) 
        self.sw1 = nn.Conv3d(latent_size, n_feats, 1, bias=True) 
        # self.sb1 = nn.Conv3d(latent_size, n_feats, 1, bias=True) 

        self.sw2 = nn.Conv3d(latent_size, n_feats, 1, bias=True) 
        # self.sb2 = nn.Conv3d(latent_size, n_feats, 1, bias=True) 
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def cal_mean_and_std(self, x):
        b, c, d, h, w = x.shape 
        x_r = x.reshape(b, c, -1) 
        
        mean = torch.mean(x_r, dim=-1)   # (b, c)
        std = torch.std(x_r, dim=-1)     # (b, c)
        
        style = torch.cat([mean, std], dim=1)  # (b, 2c)
        
        return style
        

    def forward(self, x, latent_w): 
        b, c, _, _, _ = x.shape
        

        sw1 = self.sw1(latent_w) 
        # sb1 = self.sb1(latent_w)
        res = self.convmod1(x, sw1.view(b, c))
        res = self.act(res) 
        

        sw2 = self.sw2(latent_w) 
        # sb2 = self.sb2(latent_w)    
        res = self.convmod2(res, sw2.view(b, c)) 
        
        out = x+res*self.res_scale
        

        style = self.cal_mean_and_std(out) 

        
        return out, style



class UniPET(nn.Module):
    def __init__(self, channels=64, res_num=8, latent_size=256, loss_func=nn.L1Loss()):
        super(UniPET, self).__init__()
        self.channels=channels 
        self.res_num = res_num
        
        
        self.san =SAN() 

        self.conv0 = nn.Conv3d(1,channels,3,stride=1, padding=1) 
        self.blocks = nn.ModuleList([ResModBlock(channels, kernel_size=3, latent_size=latent_size) for _ in range(res_num)])
        self.reconstruct = nn.Conv3d(channels,1,3,stride=1, padding=1) 
        self.loss_func = loss_func
        # self.smooth = Blur3D()
   
    def prepare_latent_list(self, latent_list): 
        length = len(latent_list)
        assert self.res_num >= length
        n_repeat = self.res_num // length
        n_remain = self.res_num - n_repeat*length 
        
        n_list = [n_repeat]*length 
        for i in range(n_remain):
            n_list[i] = n_list[i]+1 
        
        # import pdb 
        # pdb.set_trace()
        
        new_latent_list = []
        for i in range(length):
            for j in range(n_list[i]): 
                new_latent_list.append(latent_list[i]) 
        
        return new_latent_list


    def forward_model(self, img): 

        x = self.conv0(img)
        b, c, _, _ ,_= x.size()
        
        latent_list = self.san(x) 
        latent_list = self.prepare_latent_list(latent_list)

        style_list = []
        for i, block in enumerate(self.blocks):
            x, style = block(x, latent_list[i]) 
            style_list.append(style) 
            
        style = torch.cat(style_list, dim=1)
        
        out = self.reconstruct(x) + img

        return out, style

        

    def forward(self, lq_img, hq_img=None): 
        

        out_img, lq_style = self.forward_model(lq_img) 
        if hq_img != None:
            with torch.no_grad(): 
                _, hq_style = self.forward_model(lq_img) 

            loss_recon = self.loss_func(out_img, hq_img) 
            loss_style = self.loss_func(lq_style, hq_style) 
            
            return out_img, loss_recon, loss_style
        
        else:
            return out_img


class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3,  norm_layer=nn.InstanceNorm3d, use_sigmoid=False):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=2, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=2, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self.linear = nn.Linear(8,1)

    def forward(self, x):
        #############
        ##extract_patch
        #############
        # patches = x.unfold(2, self.patch_size, self.patch_stride).unfold(3, self.patch_size, self.patch_stride).unfold(4, self.patch_size, self.patch_stride)
        # patches = patches.reshape(-1, self.input_nc, self.patch_size, self.patch_size, self.patch_size)
        
        x = self.model(x)
        x = x.view(-1,8)
        x = self.linear(x)
        return x 
    
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
if __name__ == "__main__":
    model = UniPET() 
    print(count_parameters(model)/1e6)
