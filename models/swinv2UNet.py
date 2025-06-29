import os

import yaml

from .swintransformerv2 import SwinTransformerV2, WindowAttention, window_partition, window_reverse, DomainMemoryBanks
import torch.nn as nn
import torch
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
from .cbam import CbamModule
from torchvision import models
from itertools import chain
import math
from timm.models.layers import to_2tuple


def load_config(size, config):
    """
    从YAML文件加载模型配置
    """
    config_path = f'configs/swinv2/{size}.yaml'
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cf = yaml.safe_load(f)

    # 提取SwinV2相关配置
    model_config = {
        'embed_dim': cf['MODEL']['SWINV2']['EMBED_DIM'],
        'depths': cf['MODEL']['SWINV2']['DEPTHS'],
        'num_heads': cf['MODEL']['SWINV2']['NUM_HEADS'],
        'window_size': cf['MODEL']['SWINV2']['WINDOW_SIZE'],
        'drop_path_rate': cf['MODEL']['DROP_PATH_RATE']
    }
    return model_config

# Decoder
def swin_v2(size, img_size=256, config=None, **kwargs):
    cf = load_config(size, config)
    # 确保img_size是整数
    if isinstance(img_size, str):
        img_size = int(img_size)
        
    model = SwinTransformerV2(
        img_size=img_size,  # 现在img_size一定是整数
        patch_size=4,
        in_chans=3,
        num_classes=3,
        embed_dim=cf['embed_dim'],
        depths=cf['depths'],
        num_heads=cf['num_heads'],
        window_size=cf['window_size'],
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=cf['drop_path_rate'],
        ape=True,
        spe=False,
        patch_norm=True,
        use_checkpoint=False
    )
    return model


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            # initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]  ##
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet_swin(nn.Module):
    # Implementing only the object path
    def __init__(self, size="swinv2_tiny_window16_256", config=None, img_size=512, num_classes=2, in_channels=3,
                 pretrained=True):
        super(UperNet_swin, self).__init__()

        self.backbone = swin_v2(size=size, img_size=img_size, config=config)
        # if size.split("_")[1] in ["small", "tiny"]:
        #     feature_channels = [192, 384, 768, 768]
        # elif size.split("_")[1] in ["base"]:
        #     feature_channels = [256, 512, 1024, 1024]
        if size.split("_")[1] in ["small", "tiny"]:  # TODO feature map调整
            feature_channels = [256, 512, 1024, 1024]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [512, 1024, 2048, 2048]
        else:
            feature_channels = [768, 1536, 3072, 3072]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=16, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 分离query和key-value的生成
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # 添加门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        # 添加层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, skip):
        shortcut = x
        
        # 归一化
        x = self.norm1(x)
        skip = self.norm2(skip)
        
        B, H, W, C = x.shape
        
        # 生成query, key, value
        q = self.q(x)
        k = self.k(skip)
        v = self.v(skip)
        
        # 重塑为多头注意力格式
        q = q.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        
        # 计算门控权重
        gate = self.gate(torch.cat([x, skip], dim=-1))
        x = gate * x + (1 - gate) * skip
        
        # 投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # 残差连接
        x = x + shortcut
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None, window_size=16, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.window_size = window_size
        
        # 上采样
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
        # 如果有skip connection，添加额外的处理
        if skip_channels is not None:
            self.conv1 = md.Conv2dReLU(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )
        else:
            self.conv1 = md.Conv2dReLU(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                use_batchnorm=True,
            )

        self.cross_attn1 = CrossBlock(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True
        )
        
        self.cross_attn2 = CrossBlock(
            dim=out_channels,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=True
        )

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        # 添加输出归一化
        self.final_norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        # x是当前的解码特征（上一层的输出或初始的最深层特征）
        # skip是两组编码器的跳跃连接(x1,x2)
        x = self.up(x)
        x = self.norm1(x)
        
        if skip is not None:
            skip1, skip2 = skip
            if skip1.shape[2:] != x.shape[2:]:
                skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=True)
                skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=True)
            
            skip1 = self.norm2(skip1)
            skip2 = self.norm2(skip2)
            x = torch.cat([x, skip1], dim=1)
        
        x = self.conv1(x)
        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        if skip is not None:
            skip1 = skip1.permute(0, 2, 3, 1)
            skip2 = skip2.permute(0, 2, 3, 1)
            x = self.cross_attn1(x, skip1)
            x = self.cross_attn2(x, skip2)
        
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.final_norm(x)  # 添加最终归一化
        return x


class Decoder(nn.Module):
    def __init__(self, encoder_channels, config, size):
        super().__init__()
        encoder_channels = encoder_channels[::-1]  # 反转通道列表

        cf = load_config(size, config)
        num_heads = cf['num_heads'][::-1]
        window_size = cf['window_size']

        self.blocks = nn.ModuleList()

        for i in range(len(encoder_channels) - 1):
            # 打印当前层的通道信息
            # print(f"Decoder block {i}: in_channels={encoder_channels[i]}, "
            #       f"out_channels={encoder_channels[i + 1]}, "
            #       f"skip_channels={encoder_channels[i + 1]}")

            self.blocks.append(DecoderBlock(
                in_channels=encoder_channels[i],
                out_channels=encoder_channels[i + 1],
                skip_channels=encoder_channels[i + 1],
                window_size=window_size,
                num_heads=num_heads[i]
            ))

    def forward(self, features1, features2=None):
        features1 = features1[::-1]  # 反转特征列表
        if features2 is not None:
            features2 = features2[::-1]

        x = features1[0]  # 从最深层开始
        # print(f"Initial decoder feature shape: {x.shape}")

        for i, block in enumerate(self.blocks):
            skip1 = features1[i + 1] if i < len(features1) - 1 else None
            skip2 = features2[i + 1] if features2 is not None and i < len(features2) - 1 else None

            # if skip1 is not None:
            #     print(f"Block {i} - x shape: {x.shape}, skip shape: {skip1.shape}")

            if skip2 is not None:
                x = block(x, (skip1, skip2))
            else:
                x = block(x, (skip1, skip1))

            # print(f"Block {i} output shape: {x.shape}")

        return x


class SegmentationHead(nn.Sequential):
    # 为什么输出是[B,num_class,H,W] 分割任务是对每个像素预测其所所属的类别
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        layers = []
        # 增强特征提取
        layers.extend([
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()  # (H,W)->(4H,4W)
        ])
        super().__init__(*layers)


class unet_swin(nn.Module):
    def __init__(self, config, size="swinv2_tiny_window16_256", img_size=512):
        super().__init__()
        self.encoder = swin_v2(
            size=size,
            img_size=img_size,
            config=None
        )
        self.training_mode = True  # 添加训练模式标志

        # todo 先使用原来的
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        elif size.split("_")[1] in ["large"]:
            feature_channels = [384, 768, 1536, 1536]
        # 根据不同模型类型选择对应的通道数
        # if size.startswith('swinv2_tiny'):
        #     feature_channels = [48, 96, 192, 384]
        # elif size.startswith('swinv2_small'):
        #     feature_channels = [64, 128, 256, 512]
        # elif size.startswith('swinv2_base'):
        #     feature_channels = [96, 192, 384, 768]
        # elif size.startswith('swinv2_large'):
        #     feature_channels = [128, 256, 512, 1024]
        # else:
        #     raise ValueError(f"Unknown model size: {size}")
        self.decoder = Decoder(feature_channels, config=config, size=size)
        self.segmentation_head = SegmentationHead(
            in_channels=feature_channels[0],
            out_channels=config.num_classes+1,
            kernel_size=3,
            upsampling=4   # deocder只恢复到patch embeding之后的维度，依旧是输入维度的1/4，因此需要上采样到输入的维度
        )

        # 创建域memory banks
        self.domain_banks = DomainMemoryBanks(num_domains=5, max_size=1)    # todo 域的个数 每个域可保存的kv个数

    def train(self, mode=True):
        super().train(mode)
        self.training_mode = mode
        if mode:
            self.encoder.use_checkpoint = True
        else:
            self.encoder.use_checkpoint = False
        return self

    def forward(self, x, domain_idx, stored_kv=None):
        device = next(self.parameters()).device
        x = x.to(device)

        try:
            x1 = x[:, :3, :, :].contiguous()
            x2 = x[:, 3:, :, :].contiguous()

            # 监控特征提取
            encoder_feature1 = self.encoder.get_unet_feature(x1)
            # print(f"[DEBUG] encoder_feature1 stats: min={encoder_feature1[-1].min():.4f}, "
            #       f"max={encoder_feature1[-1].max():.4f}, "
            #       f"mean={encoder_feature1[-1].mean():.4f}")

            # 使用当前输入生成q
            decoder_output1 = self.decoder(encoder_feature1)
            x1_output = self.segmentation_head(decoder_output1)

            # 监控分数计算
            x1_softmax = F.softmax(x1_output, dim=1)
            # print(f"[DEBUG] x1_softmax stats: min={x1_softmax.min():.4f}, "
            #       f"max={x1_softmax.max():.4f}, "
            #       f"mean={x1_softmax.mean():.4f}")

            x1_score = x1_softmax.max(dim=1)[0].mean()

            # 获取x2的特征
            encoder_feature2 = self.encoder.get_unet_feature(x2)

            decoder_output = self.decoder(encoder_feature1, encoder_feature2)
            masks = self.segmentation_head(decoder_output)

            # 监控交互分数
            masks_softmax = F.softmax(masks, dim=1)
            # print(f"[DEBUG] masks_softmax stats: min={masks_softmax.min():.4f}, "
            #       f"max={masks_softmax.max():.4f}, "
            #       f"mean={masks_softmax.mean():.4f}")

            interaction_score = masks_softmax.max(dim=1)[0].mean()

            # 检查分数
            if torch.isnan(interaction_score) or torch.isnan(x1_score):
                # print("[WARNING] NaN detected in scores")
                interaction_score = torch.tensor(0.0).to(device)
                x1_score = torch.tensor(1.0).to(device)

            # 限制分数范围
            interaction_score = torch.clamp(interaction_score, 0.0, 1.0)
            x1_score = torch.clamp(x1_score, 0.0, 1.0)

            # 初始化各种损失
            memory_loss = torch.tensor(0.0).to(device)
            consistency_loss = torch.tensor(0.0).to(device)

            # 获取特征，无论是训练还是验证
            q_features, kv_features = self.encoder.extract_domain_kv(x2)   # 获取x2经过encoder后的特征

            # 添加调试信息
            print(f"[DEBUG] q_features shape: {q_features.shape}")
            print(f"[DEBUG] kv_features shape: {kv_features.shape}")

            # 分别处理每个样本的域索引
            for i in range(len(domain_idx)):
                idx = domain_idx[i].item()
                # 现在kv_features是一个[2, B, num_heads, L, head_dim]的张量
                sample_features = kv_features[:, i:i+1]  # 取出当前样本的特征

                if self.training:
                    updated = self.domain_banks.banks[f'domain{idx}'].update(
                        sample_features,
                        interaction_score
                    )
                    
                    if updated:
                        curr_memory_loss = self.domain_banks.compute_loss(sample_features, idx)
                        curr_consistency_loss = F.mse_loss(masks, x1_output)
                        
                        memory_loss += curr_memory_loss
                        consistency_loss += curr_consistency_loss
                else:
                    memory_loss += self.domain_banks.compute_loss(sample_features, idx)
                    consistency_loss += F.mse_loss(masks, x1_output)

            # 计算平均损失
            batch_size = len(domain_idx)
            memory_loss = memory_loss / batch_size  # 对比损失
            consistency_loss = consistency_loss / batch_size  # 一致性损失

            # 只在训练模式下检查梯度
            if self.training:
                with torch.no_grad():
                    # 对于非叶子张量,使用retain_grad()保留梯度
                    if decoder_output1.requires_grad:
                        decoder_output1.retain_grad()
                    if x1_output.requires_grad:
                        x1_output.retain_grad()
                    
                    if decoder_output1.grad is not None:
                        grad_norm = torch.norm(decoder_output1.grad)
                        if grad_norm > 1.0:
                            print(f"[WARNING] Large gradient in decoder_output1: {grad_norm}")

                    if x1_output.grad is not None:
                        grad_norm = torch.norm(x1_output.grad)
                        if grad_norm > 1.0:
                            print(f"[WARNING] Large gradient in x1_output: {grad_norm}")

            return masks, x1_output, interaction_score, {
                'memory_loss': memory_loss,
                'consistency_loss': consistency_loss
            }

        except Exception as e:
            print(f"[ERROR] Exception in forward pass: {str(e)}")
            print(f"[ERROR] Error occurred at tensor shape: {x.shape}")
            print(f"[ERROR] x1 shape: {x1.shape}, x2 shape: {x2.shape}")
            print(f"[ERROR] Stack trace:")
            import traceback
            traceback.print_exc()
            raise e

    def get_backbone_params(self):
        return self.encoder.parameters()

    def get_decoder_params(self):
        return self.decoder.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
