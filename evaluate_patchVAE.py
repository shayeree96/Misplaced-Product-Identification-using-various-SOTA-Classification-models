import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convPatchVAE import ConvPatchVAE
from dataset import dataset
import argparse

parser = argparse.ArgumentParser(description='VAE Digits')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--nFilters', type=int, default=128)
parser.add_argument('--nFeature', type=int, default=250)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='dataset/reconstruction/B12')
parser.add_argument('--modelname', type=str, default='trainB3_12_test12_v1')
args = parser.parse_args()

test_loader = torch.utils.data.DataLoader(
    dataset.Digits('dataset/test_exp_UPC.txt', folder=['B8'],
                   data_folder='boxes_UPC',
                   transform=transforms.Compose([transforms.Resize((128, 256)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False, num_workers=2)

vae = ConvPatchVAE(width=256, height=128, nChannels=3, hidden_size=250, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)
pretrained_model = 'models/upc_trainB3_12_test12128_250_epoch-50'
vae.loadweight_from(pretrained_model)
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)
vae.test(test_loader, save_folder=args.save_folder)

