import math

from typing import Sequence
import ipdb
import mmengine
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmcv.cnn.bricks import ConvModule
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_
import torch.nn.functional as F

from mmpretrain.models import build_2d_sincos_position_embedding

from transformers.models.mamba.modeling_mamba import MambaMixer

from mmpretrain.registry import MODELS
from mmpretrain.models.utils import build_norm_layer, resize_pos_embed, to_2tuple
from mmpretrain.models.backbones.base_backbone import BaseBackbone

import os
#配置cuda卡
torch.cuda.set_device(2)
print(torch.cuda.is_available())
print("Current CUDA device index:", torch.cuda.current_device())

def LIM(input, shift_pixel=1, head_dim=64, 
                      patch_resolution=None, with_cls_token=False):
    B, N, C = input.shape
    # 8,784,192
    assert C % head_dim == 0
    assert head_dim % 4 == 0
    if with_cls_token:
        cls_tokens = input[:, [-1], :]
        input = input[:, :-1, :]
    input = input.transpose(1, 2).reshape(
        B, 3, -1, patch_resolution[0] ,patch_resolution[1])  # [B, n_head=3, head_dim H, W]   
    # 8 3 64 28 28
    B, _, _, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, :, 0:int(head_dim*1/4), :, shift_pixel:W] = \
        input[:, :, 0:int(head_dim*1/4), :, 0:W-shift_pixel]
    output[:, :, int(head_dim/4):int(head_dim/2), :, 0:W-shift_pixel] = \
        input[:, :, int(head_dim/4):int(head_dim/2), :, shift_pixel:W]
    output[:, :, int(head_dim/2):int(head_dim/4*3), shift_pixel:H, :] = \
        input[:, :, int(head_dim/2):int(head_dim/4*3), 0:H-shift_pixel, :]
    output[:, :, int(head_dim*3/4):int(head_dim), 0:H-shift_pixel, :] = \
        input[:, :, int(head_dim*3/4):int(head_dim), shift_pixel:H, :]
    if with_cls_token:
        output = output.reshape(B, C, N-1).transpose(1, 2)
        output = torch.cat((output, cls_tokens), dim=1)
    else:
        output = output.reshape(B, C, N).transpose(1, 2)
    return output


@MODELS.register_module()
class LiMamba(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768 // 4,
                'num_layers': 8*2,
                'feedforward_channels': 768 // 2,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768//4,
                'num_layers': 12*2,
                'feedforward_channels': 768//2
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024//4,
                'num_layers': 36,
                'feedforward_channels': 1024//2
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                'embed_dims': 1280//4,
                'num_layers': 48,
                'feedforward_channels': 1280//2
            }),
    }
    OUT_TYPES = {'featmap', 'avg_featmap', 'cls_token', 'raw'}

    def __init__(self,
                 arch='base',
                 pe_type='learnable',
                 # 'forward', 'forward_reverse_mean', 'forward_reverse_gate', 'forward_reverse_shuffle_gate'
                 path_type='forward_reverse_shuffle_gate',
                 cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
                 img_size=224,
                 patch_size=8,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='avg_featmap',
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(LiMamba, self).__init__(init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)
        self.cls_position = cls_position
        self.path_type = path_type

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.num_extra_tokens = 0
        # Set cls token
        if cls_position != 'none':
            if cls_position == 'head_tail':
                self.cls_token = nn.Parameter(torch.zeros(1, 2, self.embed_dims))
                self.num_extra_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
                self.num_extra_tokens = 1
        else:
            self.cls_token = None

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pe_type = pe_type
        if pe_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        elif pe_type == 'sine':
            self.pos_embed = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                temperature=10000,
                cls_token=False)
            # TODO: add cls token
        else:
            self.pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.layers = ModuleList()
        self.gate_layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                state_size=16,
                intermediate_size=self.arch_settings.get('feedforward_channels', self.embed_dims * 2),
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                use_bias=False,
            )
            _layer_cfg.update(layer_cfgs[i])
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.layers.append(MambaMixer(_layer_cfg, i))
            if 'gate' in self.path_type:
                gate_out_dim = 2
                if 'shuffle' in self.path_type:
                    gate_out_dim = 5
                self.gate_layers.append(
                    nn.Sequential(
                        nn.Linear(gate_out_dim*self.embed_dims, gate_out_dim, bias=False),
                        nn.Softmax(dim=-1)
                    )
                )

        self.frozen_stages = frozen_stages
        self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)
        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(LiMamba, self).init_weights()

        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if 'gate' in self.path_type:
                m = self.gate_layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False
                    
    def forward(self, x):
        """ #动态卷积 + STE获取特征图:
        c = DynamicConv2d(x)
        c = STE(c) """
        """ c1, c2, c3, c4 = spm(x)  # 通过SPM模块获取c1, c2, c3, c4
        embed_dim=768//4
        level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        level_embed=level_embed.cuda()
        def _add_level_embed(c2, c3, c4):  # 定义添加层嵌入函数
          c2 = c2 + level_embed[0]  # 为c2添加嵌入
          c3 = c3 + level_embed[1]  # 为c3添加嵌入
          c4 = c4 + level_embed[2]  # 为c4添加嵌入
          return c2, c3, c4  # 返回c2, c3, c4
        c2, c3, c4 = _add_level_embed(c2, c3, c4)  # 为c2, c3, c4添加嵌入
        c = torch.cat([c2, c3, c4], dim=1)  # 将c2, c3, c4拼接 
        c = F.interpolate(c.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False)
        c = c.expand(-1, 3, -1, -1)
        x=x+c  """
        #x = feature_injection(x, c).cuda()
        #x = injector(x,c)
        B = x.shape[0]
        # Convolutional layer
        #t2
        """ model = SimpleCNN()
        model.cuda()
        c = model(x)  """
        #t3
        """ model = ResNet18(num_channels=3)
        model.cuda()
        c = model(x) """
        #t4 
        """ model = SimpleCNN2(in_channels=3)
        model.cuda()
        c = model(x) """
        
        
        """ a = nn.Parameter(torch.Tensor([0.5])).cuda()
        x = a*x + (1-a)*c """
        
        #x = x+c
        #x = self.conv(x)
        x, patch_resolution = self.patch_embed(x)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.cls_position == 'head':
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_position == 'tail':
                x = torch.cat((x, cls_token), dim=1)
            elif self.cls_position == 'head+tail':
                x = torch.cat((cls_token[:, :1], x, cls_token[:, 1:]), dim=1)
            elif self.cls_position == 'middle':
                x = torch.cat((x[:, :x.size(1) // 2], cls_token, x[:, x.size(1) // 2:]), dim=1)
            else:
                raise ValueError(f'Invalid cls_position {self.cls_position}')

        if self.pos_embed is not None:
            x = x + self.pos_embed.to(device=x.device)
        x = self.drop_after_pos(x)
        outs = []
        for i, layer in enumerate(self.layers):
            #引入q-shift操作,参数随着层数加深而减小
            with torch.no_grad():
                ratio_1_to_almost0 = (1.0 - (i  / len(self.layers)))
                #q-shift操作计算x*
            xx = LIM(x, shift_pixel=1, head_dim=64, 
                        patch_resolution = patch_resolution, with_cls_token=False)
            #print(x.shape)
            #ipdb.set_trace()
                #计算更新后的x：           
            x =  x * ratio_1_to_almost0 + xx * (1 - ratio_1_to_almost0)
            residual = x
            if 'forward' == self.path_type:
                x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
                x = layer(x)

            if 'forward_reverse_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                x = (forward_x + torch.flip(reverse_x, [1])) / 2

            if 'forward_reverse_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x
            """if 'forward_reverse_shuffle_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                mean_shuffle_x = torch.mean(shuffle_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x """
            if 'forward_reverse_shuffle_gate' == self.path_type:
                #将原本的序列、反转后的序列还有随机顺序的序列拼接到一起
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                #添加两条新路径:reverse_shuffle,shuffle_reverse,并且和之前的序列拼接到一起
                x_inputs.append(torch.flip(x[:, rand_index], [1]))
                x_inputs.append(torch.flip(x, [1])[:, rand_index])
                #将这些序列的张量按照第一个维度拼接
                x_inputs = torch.cat(x_inputs, dim=0)
                #经行预处理，归一化，初始化权重等等，为了可以正常输入到后面模型之中
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                #输入到layer层中进行训练得到输出
                x1, x2, x3, x4 = torch.chunk(x_inputs, 4, dim=2)
                #将分割好的四个通道分别送入layer层中经行运算
                x1 = layer(x1) 
                x2 = layer(x2) 
                x3 = layer(x3)
                x4 = layer(x4) 
                x_inputs = torch.cat([x1, x2,x3,x4], dim=2)
                #将输出序列按照之前的拼接分割为五个序列
                forward_x, reverse_x, shuffle_x, shuffle_reverse_x, reverse_shuffle_x = torch.split(x_inputs, B, dim=0)
                #将每个序列序列恢复到正常的顺序
                reverse_x = torch.flip(reverse_x, [1])
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                shuffle_reverse_x = shuffle_reverse_x[:, rand_index]
                reverse_shuffle_x = reverse_shuffle_x[:, rand_index]
                #计算每个路径沿着维度1的均值，用来后面生成门控信息
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                mean_shuffle_x = torch.mean(shuffle_x, dim=1)
                mean_shuffle_reverse_x = torch.mean(shuffle_reverse_x, dim=1)
                mean_reverse_shuffle_x = torch.mean(reverse_shuffle_x, dim=1)
                #生成门控信息，并且产生输出
                gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x, mean_shuffle_reverse_x, mean_reverse_shuffle_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x + gate[:, 3:4] * shuffle_reverse_x + gate[:, 4:5] * reverse_shuffle_x
            if 'forward_reverse_shuffle_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                x = (forward_x + reverse_x + shuffle_x) / 3
            x = residual + x
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            if self.cls_position == 'head':
                return x[:, 0]
            elif self.cls_position == 'tail':
                return x[:, -1]
            elif self.cls_position == 'head_tail':
                x = torch.mean(x[:, [0, -1]], dim=1)
                return x
            elif self.cls_position == 'middle':
                return x[:, x.size(1) // 2]
        patch_token = x
        if self.cls_token is not None:
            if self.cls_position == 'head':
                patch_token = x[:, 1:]
            elif self.cls_position == 'tail':
                patch_token = x[:, :-1]
            elif self.cls_position == 'head_tail':
                patch_token = x[:, 1:-1]
            elif self.cls_position == 'middle':
                patch_token = torch.cat((x[:, :x.size(1) // 2], x[:, x.size(1) // 2 + 1:]), dim=1)
        if self.out_type == 'featmap':
            B = x.size(0)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))
        

