# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from model.VisRetNet import VisRetNet
from model.resnet import ResNet


# DistilledVisionTransformer继承VisionTransformer的所有属性和方法
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, crop=False, save=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  # token

        num_patches = self.patch_embed.num_patches  # patch_size = 16 * 16
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()  # 384->1000

        trunc_normal_(self.dist_token, std=.02) # 均值为0标准差为0.02的正态分布中采样，并将采样到的值截断到指定的范围内（默认范围为 [-2, 2]）
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights) #也是初始化，和上面一样
        self.save = save # None 
        self.crop = crop # false
        self.crop_rate = 0.64 # keep rate: 0.53 352, 0.64 320, 0.79 288


    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]  # torch.Size([32, 3, 112, 616])  
        x = self.patch_embed(x)  # torch.Size([32, 266, 384])
        # torch.Size([32, 1, 384])
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)  # 随机失活
        # torch.Size([32, 268, 384]) ->
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward_features_save(self, x, indexes=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        x_shape = x.shape

        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks)-1:  # len(self.blocks)-1:
                y = blk.norm1(x) # torch.Size([32, 258, 384])
                B, N, C = y.shape  # 经过 qkv：torch.Size([32, 258, 1152])
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
                # torch.Size([3, 32, 6, 258, 64]) -> qkv ->att torch.Size([32, 6, 258, 258])
                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                # torch.Size([32, 6, 2, 256]) -> (32, 256)
                # 取cls 和 dis之间的注意力 6 * 2 *  256 作为激活图
                # 每个头的注意力，我对他关注，他对我的关注，都合起来，作为关注
                # 但是这个256是一个patch的内容，为什么可以用来作为16 * 16 个patch的关注呢
                # 激活图
                last_map = (att[:, :, :2, 2:].detach().cpu().numpy()).sum(axis=1).sum(axis=1)
                last_map = last_map.reshape(
                    [last_map.shape[0],x_shape[2] // 16, x_shape[3] // 16])

            x = blk(x)
        # last_map : (32, 16, 16)  
        for j, index in enumerate(indexes.cpu().numpy()): # (16, 16) -> (16, 16, 3)  
            plt.imsave(os.path.join(self.save, str(indexes[j].cpu().numpy()) + '.png'),
                np.tile(np.expand_dims(last_map[j]/ np.max(last_map[j]), 2), [1, 1, 3]))

        x = self.norm(x)

        return x[:, 0], x[:, 1]

    def forward_features_crop(self, x, atten):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # add nonuniform-cropping

        B = x.shape[0]
        grid_size = (x.shape[-2] // self.patch_embed.patch_size[0], x.shape[-1] // self.patch_embed.patch_size[1])
        x = self.patch_embed(x)
        # sort based on attention
        atten_reshape = torch.nn.functional.interpolate(atten.detach(), grid_size, mode='bilinear')
        order = torch.argsort(atten_reshape[:,0,:,:].reshape([B,-1]),dim=1)
        # select patches
        select_list = []
        pos_list = []
        for k in range(B):
            select_list.append(x[k,order[[k],-int(self.crop_rate*order.shape[1]):]])
            pos_list.append(torch.cat([self.pos_embed[:,:2],self.pos_embed[:,2+order[k,-int(self.crop_rate*order.shape[1]):]]],dim=1))

        x = torch.cat(select_list,dim=0)
        pos_embed = torch.cat(pos_list,dim=0)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]


    def forward(self, x, atten=None, indexes=None):
        if self.save is not None:
            x, x_dist = self.forward_features_save(x, indexes)
        elif self.crop:
            if atten is None:
                atten = torch.zeros_like(x).cuda()
            x, x_dist = self.forward_features_crop(x, atten)
        else:   # q 
            x, x_dist = self.forward_features(x)

        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        # follow the evaluation of deit, simple average and no distillation during training, could remove the x_dist
        return (x + x_dist) / 2

# 预定义模型，实例后相互隔离
@register_model
def  deit_small_distilled_patch16_224(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=384, num_classes=num_classes, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        # for key in checkpoint["model"]:
        #     print(key)


        # 通过插值，将token个数不一致的位置编码问题解决，resize the positional embedding
        # patch_size 一直是16 * 16
        # 预训练权重的patch positional embedding weight shape : torch.Size([1, 198, 384])
        weight = checkpoint["model"]['pos_embed'] # 224 / 16 = 14  # shape
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)  # 原来patch的长宽个数：14 x 14
        # 求现在要的patch的长宽个数7,38
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1]) # 7,38
        # 把非cls和dis token的其他token的positional embedding通过插值，转换为我们所需要的
        # 将原有的预训练weight展开
        matrix = weight[:, 2:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = torchvision.transforms.Resize(new_size)
        # 将权重图作为图像，认为是 batch_size,chanel,H,W 进行H、W的重塑
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        # 将cls、dis的token连接回来
        checkpoint["model"]['pos_embed'] = torch.cat([weight[:, :2, :], new_matrix], dim=1)


        # change the prediction head if not 1000
        if num_classes != 1000:
            checkpoint["model"]['head.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            checkpoint["model"]['head_dist.weight'] = checkpoint["model"]['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint["model"]['head_dist.bias'] = checkpoint["model"]['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint["model"])
        else: # in
            model.load_state_dict(checkpoint["model"])
    return model



@register_model
def res_net(pretrained=True,DEPTH = 50,REDUCTION_DIM = 2048):
    model = ResNet(DEPTH,REDUCTION_DIM)
    # 定义模型权重文件的路径
    if pretrained:
        weights_path = "model/need_dict.pth"
        # 加载模型权重
        state_dict = torch.load(weights_path)
        new_state_dict = {k.replace('encoder_q.', ''): v for k, v in state_dict.items() if k.startswith('encoder_q.')}
        model.load_state_dict(new_state_dict, strict=False)
    
    return model



@register_model
def RMT_S(args = None):
    model = VisRetNet(
        embed_dims=[64, 128, 256, 512],
        depths=[3, 4, 18, 4],
        num_heads=[4, 4, 8, 16],
        init_values=[2, 2, 2, 2],
        heads_ranges=[4, 4, 6, 6],
        mlp_ratios=[4, 4, 3, 3],
        drop_path_rate=0.15,
        chunkwise_recurrents=[True, True, True, False],
        layerscales=[False, False, False, False]
    )
    model.default_cfg = _cfg()

    checkpoint = torch.load("./RMT-S.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)


    return model



