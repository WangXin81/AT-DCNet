import torch
import os
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from models import resnet1
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d


class SiameseResnet(nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        super(SiameseResnet, self).__init__()
        self.input_nc = input_nc
        expand = 1
        if backbone == 'resnet18':
             self.resnet = resnet1.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])  # 是否应该用扩张卷积代替 2x2 步幅
        elif backbone == 'resnet34':
            self.resnet = resnet1.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = resnet1.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)  # 该上采样层将在两个维度（宽度和高度）上将输入的大小增加 2 倍。
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')  # 该上采样层会将输入的大小增加 4 倍。双线性插值
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)  # 分类器
        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            last_stage_out = 512 * expand
        elif self.resnet_stages_num == 4:
            last_stage_out = 256 * expand
        elif self.resnet_stages_num == 3:
            last_stage_out = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(last_stage_out, 32, kernel_size=(3, 3), padding=1)  # 分支内降维
        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)  # 1/2，in=3,out=64
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # 1/4
        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128
        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256
        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/8, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)  # 1/4
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BaseTransformer(SiameseResnet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,  # with_pos是否使用位置编码
                 token_len=4, if_trans=True,
                 enc_depth=1, dec_depth=1,  # 编码器和解码器的深度。
                 dim_head=8, decoder_dim_head=8,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18', show_feature_maps=False,
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BaseTransformer, self).__init__(input_nc, output_nc, backbone=backbone,
                                              resnet_stages_num=resnet_stages_num,
                                              if_upsample_2x=if_upsample_2x, )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=(1, 1),
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        self.show_Feature_Maps = show_feature_maps
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.if_trans = if_trans
        self.tokens = 0
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.token_len * 2, 32))  # 用于创建可训练的参数（权重和偏置），这些参数会在模型训练过程中自动更新
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape  # 8x32x64x64
        spatial_attention = self.conv_a(x)  # 8x4x64x64
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()  # 8x4x4096
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()  # 8x32x4096
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)  # 8x4x32
        spatial_attention = spatial_attention.view([b, self.token_len, h, w])

        return tokens, spatial_attention

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size,
                                          self.pooling_size])  # x被池化成[self.pooling_size, self.pooling_size]大小
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, (self.pooling_size, self.pooling_size))
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape  # 8x32x64x64
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')  # 8x4096x32
        x = self.transformer_decoder(x, m)  # 8x4096x32, 8x4x32->8x4096x32
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)  # 8x32x64x64
        return x

    @staticmethod
    def _forward_simple_decoder(x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
                # ,project_name, name):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)  # 8x32x64x64

        #  forward tokenzier
        if self.tokenizer:
            token1, spatial_attention1 = self._forward_semantic_tokens(x1)  # 8x4x32
            token2, spatial_attention2 = self._forward_semantic_tokens(x2)  # 8x4x32
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.if_trans:
            self.tokens = torch.cat([token1, token2], dim=1)  # 8x8x32
            self.tokens = self._forward_transformer(self.tokens)
            token1, token2 = self.tokens.chunk(2, dim=1)  # 8x4x32,8x4x32
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)  # 8x32x64x64
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)  # 8x32x256x256
        # forward small cnn
        x = self.classifier(x)  # 8x2x256x256
        if self.output_sigmoid:
            x = self.sigmoid(x)

        return x
