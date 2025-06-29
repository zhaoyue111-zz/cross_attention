import numpy as np
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple, DropPath
import math


class MemoryBank:
    def __init__(self, max_size=1, device='cuda'):
        self.max_size = max_size
        self.device = device
        self.memory = []
        self.scores = []
        
    def update(self, features, score):
        print(f"[DEBUG] MemoryBank.update called")
        print(f"[DEBUG] Input features shape: {features.shape}")
        print(f"[DEBUG] Score: {score:.4f}")
        print(f"[DEBUG] Current memory size: {len(self.memory)}")
        
        features = features[:, 0:1, ...].detach().cpu()
        score = score.detach().cpu()
        
        updated = False
        if len(self.memory) < self.max_size:
            print("[DEBUG] Memory bank not full, adding new feature")
            self.memory.append(features)
            self.scores.append(score)
            updated = True
        else:
            min_score_idx = torch.argmin(torch.tensor(self.scores))
            min_score = self.scores[min_score_idx]
            print(f"[DEBUG] Memory bank full. Min score: {min_score:.4f}, New score: {score:.4f}")
            
            if score > min_score:
                print(f"[DEBUG] Replacing feature at index {min_score_idx}")
                self.memory[min_score_idx] = features
                self.scores[min_score_idx] = score
                updated = True
            else:
                print("[DEBUG] Score too low, not updating")
        
        return updated


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    print(f"window_partition input shape: B={B}, H={H}, W={W}, C={C}, window_size={window_size}")
    
    # 确保H和W能被window_size整除
    assert H % window_size == 0, f"H {H} should be divisible by window_size {window_size}"
    assert W % window_size == 0, f"W {W} should be divisible by window_size {window_size}"
    
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DomainMemoryBanks:
    def __init__(self, num_domains=5, max_size=1):
        self.banks = {}
        for i in range(num_domains):
            self.banks[f'domain{i}'] = MemoryBank(max_size=max_size)
            
    def compute_loss(self, current_features, current_domain_idx):
        """计算跨域对比损失"""
        if not any(bank.memory for bank in self.banks.values()):
            print("[DEBUG] No features in memory banks")
            return torch.tensor(0.0).to(current_features.device)
            
        current_features = current_features.float()
        current_bank = self.banks[f'domain{current_domain_idx}']
        
        # 收集其他域的特征作为负样本
        neg_features = []
        for domain_name, bank in self.banks.items():
            if domain_name != f'domain{current_domain_idx}' and bank.memory:
                neg_features.extend(bank.memory)
                
        if not neg_features or not current_bank.memory:
            print("[DEBUG] No negative samples or current domain has no features")
            return torch.tensor(0.0).to(current_features.device)
        
        # 打印形状信息
        print(f"[DEBUG] Current features shape: {current_features.shape}")
        print(f"[DEBUG] Memory features shape: {current_bank.memory[0].shape}")
        print(f"[DEBUG] Number of negative samples: {len(neg_features)}")
        
        # 计算跨域对比损失
        pos_features = current_bank.memory[0].to(current_features.device)
        neg_features = torch.stack(neg_features).to(current_features.device)
        
        # 统一特征维度
        B = current_features.size(0)  # 应该是2
        current_feat = current_features.reshape(B, -1)  # [B, N]
        pos_feat = pos_features.reshape(B, -1)  # [B, N]
        neg_feat = neg_features.reshape(len(neg_features), B, -1)  # [num_neg, B, N]
        
        # 归一化
        current_feat_norm = F.normalize(current_feat, dim=1)  # [B, N]
        pos_feat_norm = F.normalize(pos_feat, dim=1)  # [B, N]
        neg_feat_norm = F.normalize(neg_feat, dim=2)  # [num_neg, B, N]
        
        # 计算正样本相似度
        pos_sim = torch.sum(current_feat_norm * pos_feat_norm, dim=1)  # [B]
        
        # 计算负样本相似度
        neg_sims = []
        for i in range(neg_feat_norm.size(0)):  # 遍历每个负样本
            neg_sim = torch.sum(current_feat_norm * neg_feat_norm[i], dim=1)  # [B]
            neg_sims.append(neg_sim)
        neg_sim = torch.stack(neg_sims, dim=1)  # [B, num_neg]
        
        # 打印相似度形状
        print(f"[DEBUG] Positive similarity shape: {pos_sim.shape}")
        print(f"[DEBUG] Negative similarity shape: {neg_sim.shape}")
        
        # 计算对比损失
        temperature = 0.07
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / temperature
        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()
        
        return loss


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 分别为两个输入创建q,k,v投影
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='xy'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)

        # 记录当前层的维度信息
        self.feature_dim = dim * window_size[0] * window_size[1]
        self.stage_info = f"dim{dim}_heads{num_heads}_window{window_size[0]}"

    def forward(self, x1, x2, mask=None):
        """
        Args:
            x1: 第一个输入特征 [num_windows*B, window_size*window_size, C]
            x2: 第二个输入特征 [num_windows*B, window_size*window_size, C]
            mask: 注意力掩码
        """
        B_, N, C = x1.shape
        device = x1.device

        # 从x1生成查询，从x2生成键和值
        q = self.q(x1).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.to(device)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # 应用softmax和dropout
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def get_kv_features(self, x):
        """获取输入的KV特征
        Args:
            x: 输入特征 [B, L, C]
        Returns:
            kv: KV特征 [2, B, num_heads, L, head_dim]
        """
        B, L, C = x.shape
        head_dim = C // self.num_heads
        
        kv = self.kv(x).reshape(B, L, 2, self.num_heads, head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_heads, L, head_dim]
        return kv


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 动态调整window_size，确保能整除
        if min(self.input_resolution) < window_size:
            self.window_size = to_2tuple(min(self.input_resolution))
            self.shift_size = 0
            print(f"Warning: Window size is too large, reset to {self.window_size}")

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size, L={L}, H*W={H*W}"
        
        # 动态调整window_size，确保能整除
        window_size = min(self.window_size[0], H, W)
        if H % window_size != 0 or W % window_size != 0:
            window_size = math.gcd(math.gcd(H, W), self.window_size[0])
            print(f"Warning: Reset window size to {window_size} to ensure divisibility")
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口划分
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)
        
        # 计算注意力
        attn_windows = self.attn(x_windows, x_windows)
        
        # 合并窗口
        x = window_reverse(attn_windows, window_size, H, W)
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def extra_repr(self) -> str:
            return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
                   f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def _init_respostnorm(self):
        if self.norm1 is not None:
            nn.init.constant_(self.norm1.bias, 0)
            nn.init.constant_(self.norm1.weight, 0)
        if self.norm2 is not None:
            nn.init.constant_(self.norm2.bias, 0)
            nn.init.constant_(self.norm2.weight, 0)



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, H=None, W=None):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        
        # 如果没有提供H和W，使用input_resolution
        if H is None or W is None:
            H, W = self.input_resolution
            
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建注意力块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x, H, W):
        print(f"BasicLayer input shape: {x.shape}, H={H}, W={W}")
        for blk in self.blocks:
            x = blk(x, H, W)
        print(f"BasicLayer output shape: {x.shape}")

        # 下采样
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H = (H + 1) // 2
            W = (W + 1) // 2

        return x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            blk._init_respostnorm()

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=2, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SinPositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels: The last dimension of the tensor to apply positional embedding to.
        """
        super(SinPositionalEncoding2D, self).__init__()
        # Adjust channels for 2D (splitting into x and y)
        channels = int(np.ceil(channels / 4) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))

    def forward(self, tensor):
        """
        Args:
            tensor: A 4D tensor of size (batch_size, channels, height, width)

        Returns:
            Positional Encoding Matrix of size (batch_size, channels, height, width)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        batch_size, orig_ch, height, width = tensor.shape
        pos_y = torch.arange(height, device=tensor.device).type(self.inv_freq.type())
        pos_x = torch.arange(width, device=tensor.device).type(self.inv_freq.type())

        # Generate sine and cosine embeddings for y and x
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)  # (H, 1, C)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)  # (W, C)

        # Expand and merge embeddings
        emb = torch.zeros((height, width, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_y
        emb[:, :, self.channels:] = emb_x
        emb = emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)

        return emb

class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=16, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, skip):
        B, H, W, C = skip.shape
        
        # 动态调整window_size
        window_size = min(self.window_size, H, W)
        if H % window_size != 0 or W % window_size != 0:
            window_size = math.gcd(math.gcd(H, W), self.window_size)
            print(f"CrossBlock: Reset window size to {window_size} to ensure divisibility")
        
        # 确保x和skip的空间尺寸一致
        if x.shape[1:3] != skip.shape[1:3]:
            x = F.interpolate(x.permute(0, 3, 1, 2), 
                            size=(H, W), 
                            mode='bilinear', 
                            align_corners=True).permute(0, 2, 3, 1)
        
        # 窗口划分
        x_windows = window_partition(x, window_size)  # [B*num_windows, window_size, window_size, C]
        skip_windows = window_partition(skip, window_size)
        
        # 计算注意力
        x_windows = x_windows.view(-1, window_size * window_size, C)
        skip_windows = skip_windows.view(-1, window_size * window_size, C)
        
        qkv = self.qkv(x_windows).reshape(-1, window_size * window_size, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_windows = (attn @ v).transpose(1, 2).reshape(-1, window_size * window_size, C)
        x_windows = self.proj(x_windows)
        x_windows = self.proj_drop(x_windows)
        
        # 合并窗口
        x = window_reverse(x_windows, window_size, H, W)
        return x

class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=32, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, spe=False,patch_norm=True, use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding2D(embed_dim).cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x ,y):
        print(f"SwinTransformerV2 input shape: {x.shape}")
        x = self.patch_embed(x)
        print(f"After patch_embed shape: {x.shape}")
        if self.ape:
            print(f"Position embed shape: {self.absolute_pos_embed.shape}")
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        elif self.spe:
            x = x + self.pos_embd(x)
            y = y + self.pos_embd(y)
        x = self.pos_drop(x)
        y = self.pos_drop(y)

        B, L, C = x.shape
        H, W = self.patches_resolution

        all_pseudo_labels = []  # 存储所有层的伪标签

        for layer in self.layers:
            x, H, W = layer(x, H, W)
            H, W = H // 2, W // 2

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x, all_pseudo_labels

    def extra_features(self, x ,y):
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        elif self.spe:
            x = x + self.pos_embd(x)
            y = y + self.pos_embd(y)
        x = self.pos_drop(x)
        y = self.pos_drop(y)
        feature = []

        B, L, C = x.shape
        H, W = self.patches_resolution

        for layer in self.layers:
            x = layer(x, y, H, W)
            H, W = H // 2, W // 2
            bs, n, f = x.shape
            h = int(n ** 0.5)

            feature.append(x.view(-1, h, h, f).permute(0, 3, 1, 2).contiguous())
        return feature

    def get_unet_feature(self, x):
        x = self.patch_embed(x)
        if self.ape:
            pos_embed = self.absolute_pos_embed.flatten(2).transpose(1,2)
            x = x + pos_embed
        x = self.pos_drop(x)

        B = x.shape[0]
        features = []
        
        # 获取初始特征图大小
        H = W = int(math.sqrt(x.shape[1]))
        
        # 依次通过每个 layer
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            # 重塑特征图为 [B, C, H, W] 式
            x_reshaped = x.transpose(1, 2).view(B, -1, H, W)
            print(f"Layer {i} feature shape: {x_reshaped.shape}, H={H}, W={W}")
            features.append(x_reshaped)
        
        # 确保特征尺寸是2的幂次
        for i in range(len(features)):
            h, w = features[i].shape[2:]
            target_h = 2 ** (int(math.log2(h)))
            target_w = 2 ** (int(math.log2(w)))
            if h != target_h or w != target_w:
                features[i] = F.interpolate(
                    features[i], 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=True
                )
                print(f"Resized feature {i} to: {features[i].shape}")
        
        return features

    def forward(self, x):
        x, pseudo_labels = self.forward_features(x)
        x = self.head(x)
        return x, pseudo_labels

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

    def extract_domain_kv(self, x):
        """提取输入图像的kv特征作为域风格，只使用最后一层
        Args:
            x: 输入图像 [B, C, H, W]
        Returns:
            kv: 最后一层的kv特征 [2, B, num_heads, L, head_dim]
        """
        x = self.patch_embed(x)
        
        if self.ape:
            pos_embed = self.absolute_pos_embed.flatten(2).transpose(1, 2)
            x = x + pos_embed
        elif self.spe:
            x = x + self.pos_embd(x)
        x = self.pos_drop(x)
        
        # 获取初始特征图大小
        H = W = int(math.sqrt(x.shape[1]))

        # 处理上采样的特征图
        if x.shape[1] > H * W:
            x = x[:, :H*W, :]
        
        # 遍历所有层直到最后
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                # 对于非最后一层，正常前向传播
                x, H, W = layer(x, H, W)
            else:
                # 在最后一层获取特征
                # 获取最后一个block的特征
                last_block = layer.blocks[-1]
                B, L, C = x.shape
                shortcut = x
                x = last_block.norm1(x)
                x = x.view(B, H, W, C)

                # 确保window_size是tuple
                window_size = last_block.window_size
                if isinstance(window_size, int):
                    window_size = (window_size, window_size)

                # pad feature maps
                pad_l = pad_t = 0
                pad_r = (window_size[0] - W % window_size[0]) % window_size[0]
                pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
                x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
                _, Hp, Wp, _ = x.shape

                # 窗口划分
                x_windows = window_partition(x, window_size[0])
                x_windows = x_windows.view(-1, window_size[0] * window_size[1], C)

                # 在获取KV特征之前，确保batch size一致
                B = x_windows.size(0)
                window_area = window_size[0] * window_size[1]
                target_B = (H * W) // window_area

                if B != target_B:
                    pad_size = target_B - B
                    x_windows = torch.cat([
                        x_windows,
                        x_windows[:pad_size]
                    ], dim=0)
                
                # 获取KV特征
                kv_features = last_block.attn.get_kv_features(x_windows)
                return kv_features
        
        # 如果没有进入if条件（虽然不应该发生），返回最后一层的特征
        last_layer = self.layers[-1]
        last_block = last_layer.blocks[-1]
        return last_block.attn.get_kv_features(x)









































