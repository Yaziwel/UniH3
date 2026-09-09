# Restore-rwkv: Efficient and effective medical image restoration with rwkv 
# https://github.com/Yaziwel/Restore-RWKV
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from einops import rearrange 
from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", sources=["./model/cuda/bi_wkv.cpp", "./model/cuda/bi_wkv_kernel.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])
class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())




class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()

        self.conv1x1x1 = nn.Conv3d(in_channels=dim, out_channels=dim,
                                 kernel_size=1, groups=dim, bias=False)
        self.conv3x3x3 = nn.Conv3d(in_channels=dim, out_channels=dim,
                                 kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5x5 = nn.Conv3d(in_channels=dim, out_channels=dim,
                                 kernel_size=5, padding=2, groups=dim, bias=False)

        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        self.conv5x5x5_reparam = nn.Conv3d(in_channels=dim, out_channels=dim,
                                         kernel_size=5, padding=2, groups=dim, bias=False)
        self.reparam_flag = True

    def forward_train(self, x):
        out_id   = x
        out_1x1x1  = self.conv1x1x1(x)
        out_3x3x3  = self.conv3x3x3(x)
        out_5x5x5  = self.conv5x5x5(x)
        # 按 alpha 加权融合
        out = (self.alpha[0] * out_id
             + self.alpha[1] * out_1x1x1
             + self.alpha[2] * out_3x3x3
             + self.alpha[3] * out_5x5x5)
        return out

    def reparam_5x5x5(self):
        # 将 1x1x1x1 和 3x3x3x3 的权重 pad 到 5x5x5x5，然后与原 5x5x5x5、一维 identity 加权合并
        # weight shape: [C,1,D,H,W]
        w1 = self.conv1x1x1.weight        # shape [dim,1,1,1,1]
        w3 = self.conv3x3x3.weight        # shape [dim,1,3,3,3]
        w5 = self.conv5x5x5.weight        # shape [dim,1,5,5,5]

        # 在 D/H/W 维度上 pad
        pw1 = F.pad(w1, (2,2, 2,2, 2,2))   # pad to 5×5×5
        pw3 = F.pad(w3, (1,1, 1,1, 1,1))   # pad to 5×5×5
        # identity branch: delta impulse in center of 5×5×5
        id_w = torch.zeros_like(pw1)
        # center位置置 1
        center = pw1.shape[-1] // 2
        id_w[:,:, center, center, center] = 1.0

        # 加权合并
        combined = ( self.alpha[0] * id_w
                   + self.alpha[1] * pw1
                   + self.alpha[2] * pw3
                   + self.alpha[3] * w5 )

        # 赋值给 reparam 卷积
        self.conv5x5x5_reparam.weight = nn.Parameter(combined.to(self.conv5x5x5_reparam.weight.device))


    def forward(self, x): 
        
        if self.training: 
            self.reparam_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.reparam_flag == True:
            self.reparam_5x5x5() 
            self.reparam_flag = False 
            out = self.conv5x5x5_reparam(x)
        elif self.training == False and self.reparam_flag == False:
            out = self.conv5x5x5_reparam(x)
        
        return out 

class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self.recurrence = 6 
        
        self.omnishift = OmniShift(dim=n_embd)


        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 


        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 



    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        
        d, h, w = resolution

        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        x = self.omnishift(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')    


        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v


        
        

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        
        for j in range(self.recurrence):
            mode = j % 6
            d, h, w = resolution  # resolution = (D, H, W)
        
            if mode == 0:
                # order = (d, h, w) — default flatten
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
        
            elif mode == 1:
                # order = (d, w, h)
                k = rearrange(k, 'b (d h w) c -> b (d w h) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d h w) c -> b (d w h) c', d=d, h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
                k = rearrange(k, 'b (d w h) c -> b (d h w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d w h) c -> b (d h w) c', d=d, h=h, w=w)
        
            elif mode == 2:
                # order = (h, d, w)
                k = rearrange(k, 'b (d h w) c -> b (h d w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d h w) c -> b (h d w) c', d=d, h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
                k = rearrange(k, 'b (h d w) c -> b (d h w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (h d w) c -> b (d h w) c', d=d, h=h, w=w)
        
            elif mode == 3:
                # order = (h, w, d)
                k = rearrange(k, 'b (d h w) c -> b (h w d) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d h w) c -> b (h w d) c', d=d, h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
                k = rearrange(k, 'b (h w d) c -> b (d h w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (h w d) c -> b (d h w) c', d=d, h=h, w=w)
        
            elif mode == 4:
                # order = (w, d, h)
                k = rearrange(k, 'b (d h w) c -> b (w d h) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d h w) c -> b (w d h) c', d=d, h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
                k = rearrange(k, 'b (w d h) c -> b (d h w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (w d h) c -> b (d h w) c', d=d, h=h, w=w)
        
            elif mode == 5:
                # order = (w, h, d)
                k = rearrange(k, 'b (d h w) c -> b (w h d) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (d h w) c -> b (w h d) c', d=d, h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T,
                             self.spatial_first[j] / T,
                             k, v)
                k = rearrange(k, 'b (w h d) c -> b (d h w) c', d=d, h=h, w=w)
                v = rearrange(v, 'b (w h d) c -> b (d h w) c', d=d, h=h, w=w)

        x = v
        x = sr * x
        x = self.output(x)
        return x



class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd

        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omnishift = OmniShift(dim=n_embd)
        
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)


    def forward(self, x, resolution):

        d, h, w = resolution

        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        x = self.omnishift(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')    


        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x



class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 


        self.att = VRWKV_SpatialMix(n_embd)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)



        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)


    def forward(self, x): 
        b, c, d, h, w = x.shape
        
        resolution = (d, h, w)

        # x = self.dwconv1(x) + x
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution) 
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
        
        # x = self.dwconv2(x) + x
        x = rearrange(x, 'b c d h w -> b (d h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        return x

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

class PixelUNShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUNShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        out_depth = in_depth // self.upscale_factor
        out_height = in_height // self.upscale_factor
        out_width = in_width // self.upscale_factor
        
        input_view = input.view(batch_size, channels,
                                out_depth, self.upscale_factor,
                                out_height, self.upscale_factor,
                                out_width, self.upscale_factor)
        
        shuffle_out = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return shuffle_out.view(batch_size, channels * (self.upscale_factor ** 3), out_depth, out_height, out_width)

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelUNShuffle3d(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv3d(n_feat, n_feat*4, kernel_size=3, stride=1, padding=1, bias=False),
                                  PixelShuffle3d(2))

    def forward(self, x):
        return self.body(x)


class Restore_RWKV(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 16, 
        num_blocks = [1,1,1,1], 
        num_refinement_blocks = 1, 
    ):

        super(Restore_RWKV, self).__init__() 
        
        print("What can I say, Mamba out!")

        self.patch_embed = nn.Conv3d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.encoder_level1 = nn.Sequential(*[Block(n_embd=dim) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[Block(n_embd=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[Block(n_embd=int(dim*2**2)) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[Block(n_embd=int(dim*2**3)) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=True)
        self.decoder_level3 = nn.Sequential(*[Block(n_embd=int(dim*2**2)) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(*[Block(n_embd=int(dim*2**1)) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[Block(n_embd=int(dim*2**1)) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[Block(n_embd=int(dim*2**1)) for i in range(num_refinement_blocks)])
        

        ###########################
            
        self.output = nn.Conv3d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=True) 
        self.loss_fun = nn.L1Loss()
    

    def forward(self, inp_img, label_img=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1) 

        # import pdb 
        # pdb.set_trace()
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3) 
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2) 
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img


        if label_img is not None: 

            loss = self.loss_fun(out_dec_level1, label_img)

            
            return loss

        else:
            return out_dec_level1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

# model = MedRWKV()
# num = count_parameters(model) 
# print(num/1e6)
if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='0' 
    # x=torch.zeros((1,3,513,513)).type(torch.FloatTensor).cuda() 
    
    import time 
    
    # # y = mapping(x)
    # G=IPT() 
    # G.cuda()
    # with torch.no_grad():
    #     y=G(x) 
    # # print(time.time()-since) 
    from thop import profile, clever_format
    
    x=torch.zeros((1, 1, 64, 64, 64)).type(torch.FloatTensor).cuda() 
    model = Restore_RWKV() 
    # print(model)
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.6f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))