import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
sys.path.append('/data_share/ymr/pycharm/SMSR/model')
import numpy as np
from basicsr.utils.registry import ARCH_REGISTRY
from huggingface_hub import PyTorchModelHubMixin
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from torch.jit import Final

from timm.layers import use_fused_attn



class MambaLayer(nn.Module):
    def __init__(self, indim):
        super(MambaLayer, self).__init__()
        self.indim = indim
        self.norm = nn.LayerNorm(self.indim)
        self.ss2d = SS2D(d_model=self.indim, d_state=16, expand=2., dropout=0)
        self.conv = Conv2d_BN(self.indim, self.indim, 1, 1, 0, groups=self.indim)  #(1*1卷积)
        #self.conv = nn.Conv2d(in_channels=self.indim, out_channels=self.indim, kernel_size=1, stride=1, padding=0,groups=self.indim)
        #self.mlp = ConvFFN(in_features=self.indim, hidden_features=self.indim*2, act_layer=nn.GELU, drop=0.)
        #self.norm2 = nn.LayerNorm(self.indim)
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.permute(0, 2, 3, 1)
        x1 = self.norm(x.view(B,H*W,C))
        x1 = x1.view(B,H,W,C)
        x1 =  self.ss2d(x1)
        x1 = (x1+x).permute(0,3,1,2)       #b,h,w,c
        x1 = self.conv(x1)
        #x1 = self.mlp(self.norm2(x1.view(B,H*W,C)))
        #x1 = x1.view(B,H,W,C).permute(0, 3, 1, 2)

        return x1

class ccmamba(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        self.in_dim8 = 8
        self.in_dim16 = 16
        self.in_dim32 = 32
        #self.conv8 = nn.Conv2d(in_channels=self.in_dim8, out_channels=self.in_dim8, kernel_size=1, stride=1, padding=0)
        #self.conv16 = nn.Conv2d(in_channels=self.in_dim16, out_channels=self.in_dim16, kernel_size=1, stride=1, padding=0) #8>16
        #self.conv32 = nn.Conv2d(in_channels=self.in_dim32, out_channels=self.in_dim32, kernel_size=1, stride=1, padding=0)  #16->32
        #self.conv64 = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0)      #32->64
        self.m8 = MambaLayer(indim=self.in_dim8)
        self.m16 = MambaLayer(indim=self.in_dim16)
        self.m32 = MambaLayer(indim=self.in_dim32)
        self.m64 = MambaLayer(indim=self.dim)
    def forward(self,x):

        x = x.permute(0,3,1,2)
        split_sizes = [8, 8, 16, 32]
        c_s = torch.split(x, split_sizes, dim=1)  #8，8，16，32
        x = torch.cat((c_s[0]+self.m8(c_s[0]),c_s[1]),dim=1)    #B,C,H,W
        x = torch.cat((self.m16(x),c_s[2]),dim=1)              #32
        x = torch.cat((self.m32(x), c_s[3]), dim=1)             #64
        x = self.m64(x)
        return x

class DWSconv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=True):
        super(DWSconv, self).__init__()
        #self.depthwise = nn.Conv2d(
        #    in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=bias
        #)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        #x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class dwconv(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Conv2d_BN(torch.nn.Sequential):  #计算合并conv+bn
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(                           #卷积层
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))                 #bn层，提高训练过程
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)        #初始化bn参数为1
        torch.nn.init.constant_(self.bn.bias, 0)                       #初始化偏置数为0

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5     #归并权重
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)   #(3*3的卷积)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)          #1*1的卷积
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)               #前向传播代码

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)    #权重初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)    #偏置初始化0

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5    #后来的权重参数
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,   #模型的维度
            d_state=16, #状态维度
            d_conv=3,
            expand=2.,   #维度扩展因子
            dt_rank="auto", #动态阈值的秩
            dt_min=0.001,   #动态阈值最大值和最小值
            dt_max=0.1,
            dt_init="random",  #动态阈值的初始化方式
            dt_scale=1.0,    #缩放因子
            dt_init_floor=1e-4, #动态阈值的下线
            dropout=0.,         #减少比率
            conv_bias=True,   #卷积层是否使用偏置
            bias=False,       #线性层是否使用偏置
            device=None,      #设备和数据类型
            dtype=None,       #创建网络参数的关键字参数
            #**kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        # split rate-----------
        if self.d_model == 8:
            #sr = 2/8
            sr = 4 / 8   #缩小
            #sr = 1/2
        elif self.d_model == 16:
            #sr = 2 / 8
            sr = 4 / 8
            #sr = 1/2
        elif self.d_model == 32:
            #sr = 4/8
            sr = 4 / 8
            #sr = 1/2
        elif self.d_model == 64:
            #sr = 6/8
            sr = 4 / 8
            #sr = 1/2

        self.d_model2 = int(self.d_model*sr)
        
        # ---------------------
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  #
        self.d_inner2 = int(self.expand * self.d_model2)  #

        self.dt_rank = math.ceil(self.d_model2 / 16) if dt_rank == "auto" else dt_rank
        #------局部卷积----------
        #self.rew1 = DWSconv(in_channels=d_model, out_channels=d_model)
        #self.rew2 = DWSconv(in_channels=d_model, out_channels=d_model)
        self.rewconv = RepDW(self.d_inner - self.d_inner2 )           #卷积核
        #self.rewconv = nn.Conv2d(
        #    in_channels=self.d_inner - self.d_inner2,
        #    out_channels=self.d_inner - self.d_inner2,
        #    groups=self.d_inner - self.d_inner2,  #深度可分离卷积
        #    bias=conv_bias,
        #    kernel_size=d_conv,
        #    padding=(d_conv - 1) // 2,
        #    **factory_kwargs,
        #)
        self.pool = nn.AvgPool2d(2)
        #------end--------------
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = RepDW(self.d_inner)
        #self.conv2d = nn.Conv2d(
        #    in_channels=self.d_inner,
        #    out_channels=self.d_inner,
        #    groups=self.d_inner,  #深度可分离卷积
        #    bias=conv_bias,
        #    kernel_size=d_conv,
        #    padding=(d_conv - 1) // 2,
        #    **factory_kwargs,
        #)
        self.act = nn.SiLU()  #silu激活函数

        self.x_proj = (
            nn.Linear(self.d_inner2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner2, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)  #4个线性层的维度堆叠
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner2, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner2, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn    #扫描函数

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)  #改一下吧
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4   #4个方向的扫描
        #torch.stack在指定维度上堆叠
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)  #flip 反转指定位置上的元素
        #爱因斯坦求和约定
        #einsum张量相乘和求和操作,实现后面两个张量相乘和求和的操作
        #前面的和后面的权重因子乘起来
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)   #第二个维度上分割成三部分//所以说：这里的第二个维度是通道数C
        #搞清楚mamba里面的dt_rank,d_state,d_state分别是什么？为什么要这么分？
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) #dts拆分后的东西，
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)   #计算偏置
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # 开始全局局部融合：--------------------------

        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        #-------全局融合-----------
        xlow, xhight = torch.split(x, [self.d_inner2, self.d_inner - self.d_inner2], dim=1)
        xhight = self.rewconv(xhight)

        #print(xquan.shape)
        x0 = xlow
        xlow = self.pool(xlow)
        res = x0 - F.interpolate(xlow, (H, W), mode='nearest')  # 最近邻插值

        # -----------------------------------------

        x = xlow
        y1, y2, y3, y4 = self.forward_core(x)   #b,c,h,w
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, int(H/2), int(W/2), -1)
        y = y.permute(0, 3, 1, 2).contiguous()   #(b,c,h,w)
        y = F.interpolate(y, scale_factor=2 ** (1), mode='bilinear') + res
        y = torch.cat((y,xhight), dim=1)
        y = y.permute(0, 2, 3, 1).contiguous()   #(b,h,w,c)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class HierarchicalTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads,  window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        #self.num_heads = num_heads
        #self.window_size = window_size
        #self.mlp_ratio = mlp_ratio
        #self.norm1 = norm_layer(dim)
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn_layer = ccmamba(dim=dim)
    def forward(self, x1, x_size, win_size):
        H, W = x_size
        B, L, C = x1.shape
        #x = self.norm1(x1)
        x = x1.view(B, H, W, C)
        x = self.attn_layer(x).permute(0,2,3,1).view(B,L,C) + x1   #B,H,W,C
        #x = self.norm1(x_attn.view(B,H*W,C)) + x1.view(B,H*W,C) #B,C,H,W,(56)  变成56才行
        # FFN
        x = x + self.mlp(self.norm2(x), x_size)
        return x

    #def extra_repr(self) -> str:
    #    return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
    #           f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

class BasicLayer(nn.Module):
    """ A basic Hierarchical Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, hier_win_ratios=[0.5, 1, 2, 4, 6, 8]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.win_hs = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]

        # build blocks
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                         num_heads=num_heads,
                                         window_size=(self.win_hs[i], self.win_ws[i]),
                                         mlp_ratio=mlp_ratio,
                                         drop=drop,
                                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                         norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):

        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, (self.win_hs[i], self.win_ws[i]))
            else:
                x = blk(x, x_size, (self.win_hs[i], self.win_ws[i]))
            i = i + 1

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class RHTB(nn.Module):
    """Residual Hierarchical Transformer Block (RHTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of heads for spatial self-correlation.
        base_win_size (tuple[int]): The height and width of the base window.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
        hier_win_ratios (list): hierarchical window ratios for a transformer block. Default: [0.5,1,2,4,6,8].
    """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=224, patch_size=4,
                 resi_connection='1conv', hier_win_ratios=[0.5, 1, 2, 4, 6, 8]):
        super(RHTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         base_win_size=base_win_size,
                                         mlp_ratio=mlp_ratio,
                                         drop=drop, value_drop=value_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         hier_win_ratios=hier_win_ratios)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
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

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
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

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

@ARCH_REGISTRY.register()
class SMSR(nn.Module, PyTorchModelHubMixin):

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=64, depths=[1,1,1,1,1,1], num_heads=[1,1,1,1,1,1],
                 base_win_size=[8, 8], mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=4, img_range=1., upsampler='pixelshuffledirect', resi_connection='1conv',
                 hier_win_ratios=[0.5, 1, 2, 4, 6, 8],
                 **kwargs):
        super(SMSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.base_win_size = base_win_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Hierarchical Transformer blocks (RHTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         base_win_size=base_win_size,
                         mlp_ratio=self.mlp_ratio,
                         drop=drop_rate, value_drop=value_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection,
                         hier_win_ratios=hier_win_ratios
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

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
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]

if __name__ == '__main__':
    upscale = 4
    base_win_size = [8, 8]
    height = (512 // upscale // base_win_size[0] + 1) * base_win_size[0]
    width = (512 // upscale // base_win_size[1] + 1) * base_win_size[1]

    ## HiT-SIR
    model = SMSR(upscale=4, img_size=(height, width),
                    base_win_size=base_win_size, img_range=1., depths=[1,1,1,1,1,1],
                    embed_dim=64, num_heads=[1,1,1,1,1,1], mlp_ratio=2, upsampler='pixelshuffledirect')
#原先是60
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("params: ", params_num)


