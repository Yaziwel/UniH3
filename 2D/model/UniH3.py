import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers 
from torch import einsum
from einops import rearrange
import math 
import torch.distributed as dist



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)






##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias) 

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = F.gelu(x1) 
        x = x1*x2
        x = self.project_out(x) 

        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1)) 
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1)) 
        # self.beta = nn.Parameter(torch.zeros(1, num_heads, 1, 1))


    def forward(self, x, v_adapt=None):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        if v_adapt is not None: 
            v_res = v_adapt - v
            v = (1-self.gamma)*v + self.gamma*v_adapt  
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) 

        out = (attn @ v) 
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w) 

        if v_adapt is not None:
            out = out + self.beta*v_res

        out = self.project_out(out)
        return out 
    
    


class ConvBlock(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1)
            ) 
        self.ca = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
                    nn.Sigmoid()
                    )

    def forward(self, x): 
        x = self.conv(x) 
        x = self.ca(x)*x
        return x


##########################################################################
class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(BasicBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type) 
        self.norm2 = LayerNorm(dim, LayerNorm_type) 

        self.norm3 = LayerNorm(dim, LayerNorm_type) 
        self.norm4 = LayerNorm(dim, LayerNorm_type) 
        
        self.global_att = ChannelAttention(dim, num_heads, bias) 
        self.local_conv = ConvBlock(num_feat = dim) 
        
        self.ffn1 = FeedForward(dim, ffn_expansion_factor, bias) 
        self.ffn2 = FeedForward(dim, ffn_expansion_factor, bias) 

    def forward(self, x, v_adapt=None): 
        # stx()
        x = x + self.global_att(self.norm1(x), v_adapt)
        x = x + self.ffn1(self.norm2(x)) 

        x = x + self.local_conv(self.norm3(x))
        x = x + self.ffn2(self.norm4(x))

        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super(TransformerBlock, self).__init__()
        
        self.block_list = nn.Sequential(*[BasicBlock(dim=dim, 
                                                     num_heads=num_heads, 
                                                     ffn_expansion_factor=ffn_expansion_factor, 
                                                     bias=bias, 
                                                     LayerNorm_type=LayerNorm_type, 
                                                     )
                                          for i in range(num_blocks)])
    def forward(self, x, v_adapt): 
        for blk in self.block_list:
            x = blk(x, v_adapt)
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x) 
    
class ImageUnShuffle(nn.Module):
    def __init__(self, in_feat, n_feat, scale):
        super(ImageUnShuffle, self).__init__()

        self.body = nn.Sequential(nn.PixelUnshuffle(scale),
                                  nn.Conv2d(in_feat*scale**2, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
                                  )

    def forward(self, x):
        return self.body(x) 
    


class H2M(nn.Module):
    def __init__(self, task_list, num_prompt=4096, dim=384, heads=8, momentum=0.99):
        super().__init__()

        self.task_list = list(task_list)
        self.num_tasks = len(self.task_list)
        self.task_to_idx = {task_name: idx for idx, task_name in enumerate(self.task_list)}
        self.shared_idx = self.num_tasks

        self.num_prompt = num_prompt
        self.dim = dim
        self.heads = heads
        self.momentum = momentum

        assert dim % heads == 0, "dim must be divisible by heads."

        # [T+1, L, C]
        # 0 ~ T-1: task-specific memory
        # T: shared memory
        init_tensor = torch.randn(self.num_tasks + 1, num_prompt, dim)

        self.memory = nn.Parameter(init_tensor, requires_grad=True)

        # EMA-updated HQ memory, not optimized by gradients
        self.memory_hq = nn.Parameter(torch.zeros_like(init_tensor), requires_grad=False)

        # Whether each task/shared HQ memory has been initialized
        self.register_buffer("memory_hq_initialized", torch.zeros(self.num_tasks + 1, dtype=torch.bool))

        self.gamma = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, 1, dim), requires_grad=True)

        self.norm_q = LayerNorm(dim, LayerNorm_type="WithBias")
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)

        self.q_project = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k_project = nn.Linear(dim, dim, bias=False)
        self.v_project = nn.Linear(dim, dim, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.proj_out = nn.Linear(dim, dim, bias=False)

    def _task_list_to_ids(self, task_list, device):
        """
        task_list:
            ["derain", "deblur", "derain"]
            or tensor([0, 1, 0])

        return:
            task_ids: [B]
        """
        if isinstance(task_list, torch.Tensor):
            return task_list.to(device=device, dtype=torch.long)

        task_ids = []
        for task_name in task_list:
            if task_name not in self.task_to_idx:
                raise ValueError(f"Unknown task name: {task_name}. Available tasks: {self.task_list}")
            task_ids.append(self.task_to_idx[task_name])

        return torch.tensor(task_ids, device=device, dtype=torch.long)

    @torch.no_grad()
    def _update_memory_hq(self, m2hq_out, task_ids):
        """
        m2hq_out: [B, T+1, L, C]
        task_ids: [B]

        Update:
            memory_hq[t]: only samples whose task_id == t
            memory_hq[T]: all samples in the batch
        """
        device = m2hq_out.device
        B, T_plus_1, L, C = m2hq_out.shape

        assert T_plus_1 == self.num_tasks + 1
        assert L == self.num_prompt
        assert C == self.dim

        sum_bank = torch.zeros_like(self.memory_hq)
        count_bank = torch.zeros(T_plus_1, device=device, dtype=m2hq_out.dtype)

        # Update task-specific memory
        for task_idx in range(self.num_tasks):
            mask = task_ids == task_idx

            if mask.any():
                # [num_task_samples, L, C] -> [L, C]
                sum_bank[task_idx] += m2hq_out[mask, task_idx].sum(dim=0)
                count_bank[task_idx] += mask.sum().to(m2hq_out.dtype)

        # Update shared memory with all samples
        shared_idx = self.shared_idx
        sum_bank[shared_idx] += m2hq_out[:, shared_idx].sum(dim=0)
        count_bank[shared_idx] += B

        # Synchronize across GPUs when using DDP
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_bank, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_bank, op=dist.ReduceOp.SUM)

        # EMA update
        for memory_idx in range(T_plus_1):
            if count_bank[memory_idx] > 0:
                agg = sum_bank[memory_idx] / count_bank[memory_idx]

                if not self.memory_hq_initialized[memory_idx]:
                    self.memory_hq[memory_idx].copy_(agg)
                    self.memory_hq_initialized[memory_idx] = True
                else:
                    self.memory_hq[memory_idx].mul_(self.momentum).add_((1.0 - self.momentum) * agg)

    def cross_attn(self, x_lq, x_hq=None, task_ids=None):
        """
        x_lq: [B, N, C]
        x_hq: [B, N, C] or None
        task_ids: [B] or None
        """
        B, N, C = x_lq.shape
        T_plus_1 = self.num_tasks + 1
        L = self.num_prompt

        assert C == self.dim

        # [T+1, L, C] -> [B, (T+1)*L, C]
        M = self.memory.reshape(1, T_plus_1 * L, C).expand(B, -1, -1)

        k_m = self.k_project(self.norm_k(M))
        q_skip = x_lq.clone()

        q_lq = rearrange(x_lq, "b n (h d) -> b h n d", h=self.heads)
        k_m = rearrange(k_m, "b m (h d) -> b h m d", h=self.heads)

        q_lq = F.normalize(q_lq, dim=-1)
        k_m = F.normalize(k_m, dim=-1)

        # [B, heads, N, (T+1)*L]
        lq2m_scores = torch.einsum("b h n d, b h m d -> b h n m", q_lq, k_m) * self.temperature
        lq2m_attn = torch.softmax(lq2m_scores, dim=-1)

        # Update task-specific and shared memory_hq during training
        if x_hq is not None:
            if task_ids is None:
                raise ValueError("task_ids/task_list must be provided when x_hq is not None.")

            v_hq = rearrange(x_hq, "b n (h d) -> b h n d", h=self.heads)

            with torch.no_grad():
                # [B, heads, (T+1)*L, N]
                m2hq_scores = lq2m_scores.transpose(-2, -1)
                m2hq_attn = torch.softmax(m2hq_scores, dim=-1)

                # [B, heads, (T+1)*L, d]
                m2hq_out = torch.einsum("b h m n, b h n d -> b h m d", m2hq_attn, v_hq)

                # [B, (T+1)*L, C]
                m2hq_out = rearrange(m2hq_out, "b h m d -> b m (h d)")

                # [B, T+1, L, C]
                m2hq_out = m2hq_out.reshape(B, T_plus_1, L, C)

                self._update_memory_hq(m2hq_out, task_ids)

        # In both training and inference, use the whole memory bank
        memory_hq_all = self.memory_hq.reshape(1, T_plus_1 * L, C).expand(B, -1, -1)

        v_m = self.gamma * memory_hq_all + (1.0 - self.gamma) * M
        v_m = self.v_project(self.norm_v(v_m))
        v_m = rearrange(v_m, "b m (h d) -> b h m d", h=self.heads)

        out = torch.einsum("b h n m, b h m d -> b h n d", lq2m_attn, v_m)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.proj_out(out) + q_skip * self.beta

        return out

    def forward(self, x_lq, x_hq=None, task_list=None):
        """
        Training:
            out = model(x_lq, x_hq, task_list=["derain", "deblur", ...])

        Inference:
            out = model(x_lq)

        x_lq: [B, C, H, W]
        x_hq: [B, C, H, W] or None
        task_list: list[str] or tensor([task_id_0, ...])
        """
        B, C, H, W = x_lq.shape

        assert C == self.dim

        if x_hq is not None:
            if task_list is None:
                raise ValueError("task_list must be provided when x_hq is not None.")

            task_ids = self._task_list_to_ids(task_list, x_lq.device)

            if task_ids.numel() != B:
                raise ValueError(f"task_list length must equal batch size. Got {task_ids.numel()} and {B}.")
        else:
            task_ids = None

        # LQ branch
        x_lq = self.q_dwconv(self.q_project(self.norm_q(x_lq)))
        x_lq_flat = rearrange(x_lq, "b c h w -> b (h w) c")

        # HQ branch is only used to update memory_hq
        if x_hq is not None:
            with torch.no_grad():
                x_hq = self.q_dwconv(self.q_project(self.norm_q(x_hq)))
                x_hq_flat = rearrange(x_hq, "b c h w -> b (h w) c")
        else:
            x_hq_flat = None

        out = self.cross_attn(x_lq_flat, x_hq_flat, task_ids)
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)

        return out



class IR(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 48,
        num_blocks = [2,3,3,4], 
        num_refinement_blocks = 2,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        task_list = []
    ):

        super(IR, self).__init__() 
        



        self.patch_embed_1 = ImageUnShuffle(in_feat=inp_channels, n_feat=dim, scale=1) 
        self.patch_embed_2 = ImageUnShuffle(in_feat=inp_channels, n_feat=int(dim*2), scale=2) 
        self.patch_embed_3 = ImageUnShuffle(in_feat=inp_channels, n_feat=int(dim*4), scale=4)
        self.patch_embed_4 = ImageUnShuffle(in_feat=inp_channels, n_feat=int(dim*8), scale=8) 
        self.input_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias) 
        
        
        self.hq_prompt_1 = H2M(task_list, num_prompt=128, dim=dim, heads=heads[0]) 
        self.hq_prompt_2 = H2M(task_list, num_prompt=128, dim=int(dim*2), heads=heads[1]) 
        self.hq_prompt_3 = H2M(task_list, num_prompt=128, dim=int(dim*4), heads=heads[2]) 
        self.hq_prompt_4 = H2M(task_list, num_prompt=128, dim=int(dim*8), heads=heads[3])
        
        
        
        self.encoder_level1 = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2 
        self.encoder_level2 = TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1])
        
        self.down2_3 = Downsample(int(dim*2)) ## From Level 2 to Level 3 
        self.encoder_level3 = TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2])

        self.down3_4 = Downsample(int(dim*4)) ## From Level 3 to Level 4 
        self.latent = TransformerBlock(dim=int(dim*8), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[3])
        
        self.up4_3 = Upsample(int(dim*8)) ## From Level 4 to Level 3
        self.reduce_chan_decoder_level3 = nn.Conv2d(int(dim*8), int(dim*4), kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerBlock(dim=int(dim*4), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2])


        self.up3_2 = Upsample(int(dim*4)) ## From Level 3 to Level 2
        self.reduce_chan_decoder_level2 = nn.Conv2d(int(dim*4), int(dim*2), kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerBlock(dim=int(dim*2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1])
        
        self.up2_1 = Upsample(int(dim*2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_decoder_level1 = nn.Conv2d(int(dim*2), dim, kernel_size=1, bias=bias)
        self.decoder_level1 = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0])
        
        self.refinement = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_refinement_blocks)
        self.reduce_refinement = nn.Conv2d(int(dim*3), dim, kernel_size=1, bias=bias)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias) 

    def forward(self, inp_img, label_img=None, inp_task_list=None): 

        
        inp_scale_1 = self.patch_embed_1(inp_img) 
        inp_scale_2 = self.patch_embed_2(inp_img) 
        inp_scale_3 = self.patch_embed_3(inp_img) 
        inp_scale_4 = self.patch_embed_4(inp_img) 
        
        if label_img is not None: 
            with torch.no_grad():
                hq_scale_1 = self.patch_embed_1(label_img) 
                hq_scale_2 = self.patch_embed_2(label_img) 
                hq_scale_3 = self.patch_embed_3(label_img) 
                hq_scale_4 = self.patch_embed_4(label_img) 
        else:
            hq_scale_1 = None
            hq_scale_2 = None
            hq_scale_3 = None
            hq_scale_4 = None 
        
        hq_enc_level1 = self.hq_prompt_1(inp_scale_1, hq_scale_1, inp_task_list) 
        hq_enc_level2 = self.hq_prompt_2(inp_scale_2, hq_scale_2, inp_task_list) 
        hq_enc_level3 = self.hq_prompt_3(inp_scale_3, hq_scale_3, inp_task_list) 
        hq_enc_level4 = self.hq_prompt_4(inp_scale_4, hq_scale_4, inp_task_list)
        
        
        
        inp_enc_level1 = self.input_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1, hq_enc_level1) 

        
        inp_enc_level2 = self.down1_2(out_enc_level1) 
        out_enc_level2 = self.encoder_level2(inp_enc_level2, hq_enc_level2) 


        inp_enc_level3 = self.down2_3(out_enc_level2) 
        out_enc_level3 = self.encoder_level3(inp_enc_level3, hq_enc_level3) 


        inp_enc_level4 = self.down3_4(out_enc_level3)      
        latent = self.latent(inp_enc_level4, hq_enc_level4) 
        
                        
        inp_dec_level3 = self.up4_3(latent) 
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, inp_scale_3) 
        


        inp_dec_level2 = self.up3_2(out_dec_level3) 
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, inp_scale_2) 



        inp_dec_level1 = self.up2_1(out_dec_level2) 
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1) 
        inp_dec_level1 = self.reduce_chan_decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, inp_scale_1)
        
        out_dec_level1_refine = self.refinement(out_dec_level1, inp_scale_1) 
        
        out = torch.cat([out_enc_level1, out_dec_level1, out_dec_level1_refine], dim=1) 
        out = self.reduce_refinement(out)
        out = self.output(out) + inp_img

        return out



class UW_Estimator(nn.Module):
    def __init__(self, in_dim=48, out_dim=1, hidden_dim=48, task_list=[], mode="log"):
        super(UW_Estimator, self).__init__() 
        task_num = len(task_list)
        if task_num!=0: 
            if mode == "exp":
                self.task_embed = nn.Parameter(torch.zeros(task_num, requires_grad=True)) 
            elif mode == "log":
                self.task_embed = nn.Parameter(torch.ones(task_num, requires_grad=True)) 
            else:
                raise ValueError(f"Invalid mode '{mode}'. Supported options are 'exp' or 'log'.")
            self.task_idx_dict = {task: index for index, task in enumerate(task_list)}
        
        self.un_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1, bias=True), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True), 
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1, bias=True),
        ) 
    def forward(self, condition, inp_task_list):
        # First linear transformation 
        uncertainty_img = self.un_conv(condition) 
        uncertainty_weight = uncertainty_img.mean(dim=(1,2,3)) 
        if len(inp_task_list)!=0:
            indices = [self.task_idx_dict[task] for task in inp_task_list] 
            task_bias = self.task_embed[indices] 
            uncertainty_weight = uncertainty_weight + task_bias
            
        return uncertainty_weight

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, dim=48, task_list = [], mode="log"):
        super(AutomaticWeightedLoss, self).__init__() 
        self.uw_estimator = UW_Estimator(in_dim=3, out_dim=1, hidden_dim=dim, task_list=task_list, mode=mode) 
        self.mode = mode

    def forward(self, inp_img, out_img, label_img, losses, inp_task_list = []): 
        out_img = out_img.clone().detach() 
        diff1 = torch.abs(inp_img-label_img) 
        diff2 = torch.abs(inp_img-out_img) 
        diff3 = torch.abs(out_img-label_img)
        condition = torch.cat([diff1, diff2, diff3], dim=1)
        loss_sum = 0 
        uncertainty_weight = self.uw_estimator(condition, inp_task_list)
        for i, loss in enumerate(losses): 
            if self.mode=="exp":
                loss_sum += torch.exp(-uncertainty_weight[i]) * loss + uncertainty_weight[i] 
            elif self.mode=="log":
                loss_sum += 1 / (uncertainty_weight[i] ** 2 + 1e-8) * loss + torch.log(uncertainty_weight[i] ** 2 + 1) 
            else: 
                raise ValueError(f"Invalid mode '{self.mode}'. Supported options are 'exp' or 'log'.")
        return loss_sum/len(losses)


class UniH3(nn.Module):
    def __init__(self, loss_fun = nn.L1Loss(), task_list = []):
        super(UniH3, self).__init__() 
        self.loss_fun = nn.L1Loss(reduction='none') 
        
        
        self.IR_Model = IR(task_list = task_list) 
        
        self.UW_loss = AutomaticWeightedLoss(task_list = task_list)



    def forward(self, inp_img, label_img=None, inp_task_list=[]): 


        
        out_img  = self.IR_Model(inp_img, label_img, inp_task_list=inp_task_list)
        
        if label_img is not None: 

            loss = self.loss_fun(out_img, label_img).view(inp_img.shape[0], -1).mean(dim=1)
            loss_weighted = self.UW_loss(inp_img, out_img, label_img, loss, inp_task_list) 
            
            return loss_weighted

        else:
            return out_img





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='0' 

    import time 
    from thop import profile, clever_format
    

    
    modality_list = ["PET", "CT", "MRI", "OCT", "Pathology", "Ultrasound", "X-ray"] 
    
    x=torch.zeros((1, 1, 128, 128)).type(torch.FloatTensor).cuda() 
    model = IR(task_list=modality_list) 
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