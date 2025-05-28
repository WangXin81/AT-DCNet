import torch
from torch.nn import init
from models import (BIT, DMINet, ICIFNet, AT_DCNet, SiamUnet_conc, SiamUnet_diff, Unet, DTCDSCN, MSCD)
from models.ChangeFormer_my import ChangeFormerV5, ChangeFormerV6



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights."""

    def init_func(m):  # define the initialization function
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (class_name.find('Conv') != -1 or class_name.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif class_name.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, gpu_ids, init_type='normal', init_gain=0.02):
    """Initialize a network,Return an initialized network."""
    str_ids = gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id1 = int(str_id)
        if id1 >= 0:
            gpu_ids.append(id1)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, gpu_ids, init_type='normal', init_gain=0.02):
    if args.net_G == 'base_resnet18':
        net = BIT.SiameseResnet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4_dd8_dedim8':
        net = BIT.BaseTransformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                                  with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8,show_feature_maps= False)

    elif args.net_G == 'DMINet':
        net = DMINet.DMINet(show_feature_maps=False)

    elif args.net_G == 'ICIFNet':
        net = ICIFNet.ICIFNet(pretrained=False)

    elif args.net_G == 'AT_DCNet':
        net = AT_DCNet.CDNet( show_feature_maps=True, pool_size = 4, depth=1)

    elif args.net_G == 'SiamUnet_conc':
        net = SiamUnet_conc.SiamUnet_conc(input_nbr=3, label_nbr=2)

    elif args.net_G == 'SiamUnet_diff':
        net = SiamUnet_diff.SiamUnet_diff(input_nbr=3, label_nbr=2)

    elif args.net_G == 'FC-EF':
        net = Unet.Unet(input_nbr=3, label_nbr=2)

    elif args.net_G == 'ChangeFormerV5':
        net = ChangeFormerV5(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'ChangeFormerV6':
        net = ChangeFormerV6(embed_dim=args.embed_dim) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == 'DTCDSCN':
        net = DTCDSCN.CDNet34(3)

    elif args.net_G == 'MSCD':
        net = MSCD.MSCDNet_v2(3, 1)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, gpu_ids, init_type, init_gain)
