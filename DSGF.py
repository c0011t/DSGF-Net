import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
# selective_scan from mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto",
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

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

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        
        # 获取四个方向的输出
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        
        # 合并四个方向的输出
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x):
        mean = x.mean((2,3), keepdim=True)
        std = x.std((2,3), keepdim=True)
        return (x - mean) / (std + self.eps) * self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)

class MSCA(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.b3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.b7 = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.proj = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x3 = self.b3(x)
        x5 = self.b5(x)
        x7 = self.b7(x)
        return self.proj(torch.cat([x3, x5, x7], dim=1))


class MDSS(nn.Module):
    def __init__(self, dim=512, d_state=32, d_conv=3, expand=4, drop_path=0.1):
        super().__init__()

        self.input_branch = nn.Linear(dim, dim)

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.linear1 = nn.Linear(dim, dim)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.ss2d = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.linear2 = nn.Linear(dim, dim)

        self.linear_side = nn.Linear(dim, dim)
        self.msca = MSCA(dim)

        self.linear_final = nn.Linear(dim, dim)

        self.drop_path = DropPath(drop_path)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        input_branch_out = self.input_branch(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_ln = self.norm1(x.permute(0, 2, 3, 1))

        side = self.linear_side(x_ln)
        side = side.permute(0, 3, 1, 2)
        side = self.msca(side)

        main = self.linear1(x_ln).permute(0, 3, 1, 2)
        main = self.dwconv(main)

        main = self.ss2d(main.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        main = main.permute(0, 2, 3, 1)
        main = self.norm2(main)
        main = self.linear2(main).permute(0, 3, 1, 2)

        fused = main * side
        fused = self.linear_final(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        fused = fused + input_branch_out
        out = identity + self.drop_path(self.alpha * fused)
        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    Works for any tensor shape. Assumes width and height are both even or both odd.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DWConv3x3BNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, stride, groups):
        super(DWConv3x3BNReLU, self).__init__(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True),
        )


class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channel, out_channel, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channel = in_channel // divide
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=mid_channel),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=out_channel),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = torch.flatten(out, start_dim=1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x


class GhostModule(nn.Module):
    def __init__(self, in_channel, out_channel, s=2, kernel_size=1, stride=1, use_relu=True):
        super(GhostModule, self).__init__()
        intrinsic_channel = out_channel // s
        ghost_channel = intrinsic_channel * (s - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intrinsic_channel, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(intrinsic_channel),
            nn.ReLU(inplace=True) if use_relu else nn.Sequential()
        )

        self.cheap_op = DWConv3x3BNReLU(in_channel=intrinsic_channel, out_channel=ghost_channel, stride=stride,
                                        groups=intrinsic_channel)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_op(x1)
        out = torch.cat([x1, x2], dim=1)
        return out


class GhostBottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        self.bottleneck = nn.Sequential(
            GhostModule(in_channel=in_channel, out_channel=mid_channel, use_relu=True),
            DWConv3x3BNReLU(in_channel=mid_channel, out_channel=mid_channel, stride=stride,
                            groups=mid_channel) if self.stride > 1 else nn.Sequential(),
            SqueezeAndExcite(in_channel=mid_channel, out_channel=mid_channel) if use_se else nn.Sequential(),
            GhostModule(in_channel=mid_channel, out_channel=out_channel, use_relu=False)
        )

        if self.stride > 1:
            self.shortcut = DWConv3x3BNReLU(in_channel=in_channel, out_channel=out_channel, stride=stride, groups=1)
        else:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.bottleneck(x)
        residual = self.shortcut(x)
        out += residual
        return out

class EGE(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, use_se):
        super(EGE, self).__init__()
        self.conv_g =  nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch_g = nn.BatchNorm2d(ch_out)
        self.relu_g = nn.ReLU(inplace=True)
        self.Ghost1 = GhostBottleneck(ch_out, 2*ch_out, ch_out, kernel_size= kernel_size, stride= stride,use_se= use_se )
        self.Ghost2 = GhostBottleneck(ch_out, 2*ch_out, ch_out, kernel_size= kernel_size, stride= stride,use_se= use_se )

        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual_g = self.Conv_1x1(x)
        
        x_conv_ghost = self.conv_g(x)
        x_conv_ghost = self.batch_g(x_conv_ghost)
        x_conv_ghost = self.relu_g(x_conv_ghost)

        x_conv_ghost = residual_g + x_conv_ghost

        x_out = self.Ghost1(x_conv_ghost)
        x_out = self.Ghost2(x_out)
        return x_out

class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.Conv_1x1(x)
        x = self.conv(x)
        x = residual + x
        x = self.relu(x)
        return x

class CGF(nn.Module):

    def __init__(self, channel):
        super(CGF, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.w1 = nn.Parameter(torch.Tensor(channel, channel))
        self.w2 = nn.Parameter(torch.Tensor(channel, channel))
        
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        l = self.gap(x).view(b, c) 
        l_diff = l.unsqueeze(1) - l.unsqueeze(2) 
        A = torch.sigmoid(l_diff) 
        I = torch.eye(c, device=x.device).unsqueeze(0)
        A_hat = A + I
        D_hat = torch.sum(A_hat, dim=2) + 1e-5
        D_inv_sqrt = torch.pow(D_hat, -0.5)
        D_mat = torch.diag_embed(D_inv_sqrt)
        A_norm = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        H_in = x.view(b, c, -1) 
        H1 = torch.bmm(A_norm, H_in)
        H1 = torch.matmul(H1.permute(0, 2, 1), self.w1).permute(0, 2, 1)
        H1 = self.relu(H1)
        H2 = torch.bmm(A_norm, H1)
        H2 = torch.matmul(H2.permute(0, 2, 1), self.w2).permute(0, 2, 1)
        x_gcn = H2.view(b, c, h, w)
        out = x + self.gamma * x_gcn
        return out

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=scale_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, input):
        x = self.up(input)
        return x

class DSGF(nn.Module):
    def __init__(self, img_ch, output_ch):
        super(DSGF, self).__init__()

        self.out_size2 = (32, 32)
        self.out_size1 = (64, 64)

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.Conv1 = EGE(img_ch, 64, 3, 1, True)
        self.Conv2 = EGE(64, 128, 3, 1, True)
        self.Conv3 = EGE(128, 256, 3, 1, True)
        self.Conv4 = EGE(256, 512, 3, 1, True)

        self.input_proj1 = nn.Conv2d(img_ch, 64, kernel_size=1, padding=0, bias=False)
        self.input_proj2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False)
        self.input_proj3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, bias=False)
        self.input_proj4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False)

        self.Dconv1 = MDSS(dim=64,  d_state=16, expand=2, drop_path=0.02)
        self.Dconv2 = MDSS(dim=128, d_state=16, expand=2, drop_path=0.05)
        self.Dconv3 = MDSS(dim=256, d_state=32, expand=3, drop_path=0.05)
        self.Dconv4 = MDSS(dim=512, d_state=32, expand=4, drop_path=0.05)

        self.Up3 = up_conv(512, 256)
        self.Up_conv3 = conv_block(512, 256)

        self.Up2 = up_conv(256, 128)
        self.Up_conv2 = conv_block(256, 128)

        self.Up1 = up_conv(128, 64)
        self.Up_conv1 = conv_block(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.cgf3 = CGF(512)
        self.cgf2 = CGF(256)
        self.cgf1 = CGF(128)

        self.dsv3 = UnetDsv3(in_size=256, out_size=128, scale_factor=self.out_size2)
        self.dsv2 = UnetDsv3(in_size=128, out_size=64, scale_factor=self.out_size1)

    def forward(self, x):
        x1 = self.Conv1(x)
        y1 = self.Dconv1(self.input_proj1(x))
        e1 = x1 + y1

        Maxpool1 = self.Maxpool(e1)

        x2 = self.Conv2(Maxpool1)
        y2 = self.Dconv2(self.input_proj2(Maxpool1))
        e2 = x2 + y2

        Maxpool2 = self.Maxpool(e2)

        x3 = self.Conv3(Maxpool2)
        y3 = self.Dconv3(self.input_proj3(Maxpool2))
        e3 = x3 + y3

        Maxpool3 = self.Maxpool(e3)

        x4 = self.Conv4(Maxpool3)
        y4 = self.Dconv4(self.input_proj4(Maxpool3))
        e4 = x4 + y4

        d3 = self.Up3(e4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = d3 + self.cgf3(d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = d2 + self.cgf2(d2)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = d1 + self.cgf1(d1)
        d1 = self.Up_conv1(d1)

        F3 = self.dsv3(d3)
        F2 = 0.3 * F3 + 0.7 * d2
        F1 = self.dsv2(F2)
        Fout = 0.3 * F1 + 0.7 * d1

        d0 = self.Conv_1x1(Fout)

        return d0


if __name__ == '__main__':

    net = DSGF(1,2)
    in1 = torch.randn((64,1,64,64))
    out1 = net(in1)
    print(out1.size())
