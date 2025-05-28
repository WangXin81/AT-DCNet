import torch
import os
from torch import nn, softmax
from models.help_funcs import Residual, PreNorm, TwoLayerConv2d, Residual2, PreNorm2


class DeformConv2d(nn.Module):  # 可变形卷积
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # 保证了所有的采样位置都在特征图的合法范围内， .long向下取整，得到坐标的左上方网格点坐标
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()  # x,y均+1，得到右下方网格点坐标
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 取左上的横坐标，右下的纵坐标，得到左下方网格点坐标
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 取右下的横坐标，左上的纵坐标，得到右上方网格点坐标
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)],
                      dim=-1)  # 带小数的坐标
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))  # 左上点的权值
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))  # 右下点的权值
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))  # 左下点的权值
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))  # 右上点的权值
        x_q_lt = self._get_x_q(x, q_lt, N)  # 找到索引位置对应的x输入图中的值， 2N变成N，插入通道C维度
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # 广播机制.unsqueeze(dim=1)加入通道C维度
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt  # 双线性插值，计算4邻域像素点的加权和，权重则根据它与插值点横、纵坐标的距离来设置

        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

    def _get_p_n(self, N, dtype):  # 卷积核的网格坐标
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):  # 输入特征图中的网格坐标
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):  # p就是加入偏移量后的输入网络坐标
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y     计算整数索引， 2N变成N
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


class BasicBlock(nn.Module):   # 残差单元
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basicblook = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            identity = self.conv1(x)
        else:
            identity = x
        out = self.basicblook(x)
        out += identity
        out = self.relu(out)

        return out

#  局部分支
class LocalBranch(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(LocalBranch, self).__init__()
        self.basicblook1 = BasicBlock(in_channels, out_channels, stride)
        self.basicblook2 = BasicBlock(out_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.deformConv = DeformConv2d(out_channels, out_channels, 3)

    def forward(self, x):
        out = self.basicblook1(x)
        out = self.basicblook2(out)

        identity = out
        out = self.conv2(out)
        out = self.bn1(out)
        out = self.deformConv(out)   # 可变形卷积层
        out += identity
        out = self.relu(out)

        return out

# 自适应多头自注意力
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=True, pool=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        self.act = nn.GELU()
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(pool)  # 自适应平均池化固定输出nxn
            self.pool2 = nn.AdaptiveMaxPool2d(pool)
            self.conv1 = nn.Conv2d(2*dim, dim, 1,1)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        H = kwargs.get('H', None)
        W = kwargs.get('W', None)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = self.act(x)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.conv1(torch.cat([self.pool(x_), self.pool2(x_)], 1))
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# 自适应多头交叉注意力
class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=True, pool=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        self.act = nn.GELU()
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(pool)  # 自适应平均池化固定输出nxn
            self.pool2 = nn.AdaptiveMaxPool2d(pool)
            self.conv1 = nn.Conv2d(2*dim, dim, 1,1)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x1, x2, **kwargs):
        B, N, C = x1.shape
        H = kwargs.get('H', None)
        W = kwargs.get('W', None)
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x1.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                x_ = self.act(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                x_ = self.act(x1)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.conv1(torch.cat([self.pool(x_), self.pool2(x_)], 1))
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=512):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

# 多层感知机MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        H = kwargs.get('H', None)
        W = kwargs.get('W', None)
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 自适应Transformer模块
class myTransformer(nn.Module):
    def __init__(self, dim, depth, heads, sr_ratio, linear, pool_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, num_heads=heads, attn_drop= dropout, proj_drop=dropout, sr_ratio=sr_ratio, linear=linear, pool=pool_size))),
                Residual(PreNorm(dim, Mlp(dim, dim*4, dropout=dropout)))
            ]))

    def forward(self, x, H, W):
        for attn, ff in self.layers:
            x = attn(x, H=H, W=W)
            x = ff(x, H=H, W=W)
        return x

# 自适应Cross-Transformer模块
class myTransformer1(nn.Module):
    def __init__(self, dim, depth, heads, sr_ratio, linear, pool_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, num_heads=heads, attn_drop= dropout, proj_drop=dropout, sr_ratio=sr_ratio, linear=linear, pool=pool_size))),
                Residual(PreNorm(dim, Mlp(dim, dim*4, dropout=dropout)))
            ]))

    def forward(self, x1, x2, H, W):
        for attn, ff in self.layers:
            x = attn(x1,x2,  H=H, W=W)
            x = ff(x, H=H, W=W)
        return x

# 卷积嵌入模块
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, image_size, emb_dropout = 0.):
        super(PatchEmbedding, self).__init__()
        self.patch_embendding = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.position_embeddings = nn.Parameter(torch.zeros(1, image_size*image_size, out_channels))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.patch_embendding(x)
        _, _, H, W = x.shape
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, H, W

# 全局分支
class GlobalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, image_size, pool_size, stride, depth , sr_ratio = (8 ,4, 2), linear = True, dropout = 0.):
        super(GlobalBranch, self).__init__()
        self.transformer = myTransformer(out_channel, depth, 8, sr_ratio, linear, pool_size,  dropout)
        self.patch_embeddings = PatchEmbedding(in_channel, out_channel, stride, stride, 0, image_size)

    def forward(self, x):
        x, H, W = self.patch_embeddings(x)
        B, N, C = x.shape
        t = self.transformer(x, H, W)
        x = t.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

# 卷积块注意力模块
class CBAM(nn.Module):#通道注意力加空间注意力模块
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

# 局部-全局融合模块
class LocalGlobalFuse(nn.Module):
    def __init__(self, in_channel, out_channel, image_size, pool_size, depth):
                 # , depth, sr_ratio, linear, pool_size, dropout):
        super(LocalGlobalFuse, self).__init__()
        self.depth = depth
        self.local = LocalBranch(in_channel, out_channel, 2)
        self.globalBranch = GlobalBranch(in_channel, out_channel, image_size, pool_size, 2, depth)
        self.CBAM = CBAM(out_channel)

    def forward(self, x):
        x_local = self.local(x)
        x_global = self.globalBranch(x)

        x_ = x_local + x_global
        x = self.CBAM(x_)
        return x

# 基于转置卷积的上采样模块
class MyTransposeConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride_2 = 1,output_padding_2 = 0):
        super(MyTransposeConvNet, self).__init__()
        # 定义第一个转置卷积层
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 定义批处理归一化层
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        # 定义第二个转置卷积层
        self.conv_transpose2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride_2, padding=1, output_padding = output_padding_2)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 第一个转置卷积层
        x = self.conv_transpose1(x)
        # 批处理归一化
        x = self.batch_norm1(x)
        # ReLU激活函数
        x = self.relu(x)
        # 第二个转置卷积层
        x = self.conv_transpose2(x)
        x = self.batch_norm2(x)

        return x


# 高阶特征交互模块
class HFI(nn.Module):
    def __init__(self, dim, pool_size, depth):
        super(HFI,self).__init__()
        self.depth = depth
        self.transformer = myTransformer1(dim, 1, 8,  1, True, pool_size, 0.0)

    def forward(self, x1, x2):
        for i in range(self.depth):
            B, C, H, W = x1.shape
            t1_16 = x1.flatten(2).transpose(1, 2)
            t2_16 = x2.flatten(2).transpose(1, 2)
            T1_16 = self.transformer(t1_16, t2_16, H, W)
            x1 = T1_16.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            T2_16 = self.transformer(t2_16, t1_16, H, W)
            x2 = T2_16.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x1, x2

# 差异特征提取模块
class DEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels*5, out_channels, 1, 1, 0)

    def forward(self, x1, x2):
        x_abs = torch.abs(x1 - x2)
        x_abs_1 = torch.abs(x_abs - x1)
        x_abs_2 = torch.abs(x_abs - x2)
        X = torch.cat([x1, x2, x_abs,  x_abs_1,  x_abs_2], 1)
        X = self.conv1(X)
        return X

# AT-DCNet方法
class CDNet(nn.Module):
    def __init__(self, show_feature_maps, pool_size, depth):
        super(CDNet, self).__init__()
        self.OverlapPatchEmbed =nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True) )
        self.LGF1 = LocalGlobalFuse(32, 64, 64, pool_size, 1)
        self.LGF2 = LocalGlobalFuse(64, 128, 32, pool_size, 1)
        self.LGF3 = LocalGlobalFuse(128, 256, 16, pool_size, 1)
        self.HFI = HFI(256, 4, depth)
        self.DEM1 = DEM(32, 32)
        self.DEM2 = DEM(64, 64)
        self.DEM3 = DEM(128, 128)
        self.DEM4 = DEM(256, 256)

        self.TransConv1 = MyTransposeConvNet(256, 128)
        self.TransConv2 = MyTransposeConvNet(256, 64)
        self.TransConv3 = MyTransposeConvNet(128, 32)
        self.TransConv4 = MyTransposeConvNet(64, 32)
        self.conv1 = nn.Conv2d(256, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(128, 128, 1, 1, 0)
        self.conv3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels = 2)

    def forward(self, x1, x2):
        # 孪生特征提取器
        x1_2 = self.OverlapPatchEmbed(x1)
        x1_4 = self.LGF1(x1_2)
        x1_8 = self.LGF2(x1_4)
        x1_16 = self.LGF3(x1_8)
        #
        x2_2 = self.OverlapPatchEmbed(x2)
        x2_4 = self.LGF1(x2_2)
        x2_8 = self.LGF2(x2_4)
        x2_16 = self.LGF3(x2_8)

        #高阶特征交互模块
        T1_16, T2_16 = self.HFI(x1_16, x2_16)

        # 差异特征提取模块
        X_16 = self.DEM4(T1_16, T2_16)
        X_8 = self.DEM3(x1_8, x2_8)
        X_4 = self.DEM2(x1_4, x2_4)
        X_2 = self.DEM1(x1_2, x2_2)

        # 多尺度融合解码器
        out_8 = self.TransConv1(X_16)
        out_4 = self.TransConv2(self.conv1(torch.cat([X_8, out_8], 1)))
        out_2 = self.TransConv3(self.conv2(torch.cat([X_4, out_4], 1)))
        out = self.TransConv4(self.conv3(torch.cat([out_2, X_2],1)))

        # 分类器
        out = self.classifier(out)

        return out

