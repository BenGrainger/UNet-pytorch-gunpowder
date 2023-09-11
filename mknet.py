
import torch
from unet import UNet, ConvPass

def create_affinity_model(in_channels, num_fmaps, fmap_inc_factor, downsample_factors, lr):

    # create unet
    unet = UNet(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors)

    # add an extra convolution to get from 12 feature maps to 3 (affs in x,y,z)
    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=num_fmaps, out_channels=3, kernel_sizes=[[1, 1, 1]], activation='Sigmoid'))

    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=(0.95,0.999))
    
    return model, optimizer