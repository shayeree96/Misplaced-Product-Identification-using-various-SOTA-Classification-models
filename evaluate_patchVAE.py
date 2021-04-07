import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from convPatchVAE import ConvPatchVAE
import dataset
import argparse

parser = argparse.ArgumentParser(description='VAE Digits')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--nFilters', type=int, default=256)
parser.add_argument('--nFeature', type=int, default=32)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='eval/reconstruction/')
parser.add_argument('--modelname', type=str, default='models/trainB3_12256_32_epoch-10')
parser.add_argument('--data_list', default='testlist.txt')
parser.add_argument('--data_folder', default='B6')
args = parser.parse_args()

test_loader = torch.utils.data.DataLoader(dataset.Digits(args.data_list, transform=transforms.Compose([transforms.Resize((112, 100)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False, num_workers=2)

vae = ConvPatchVAE(width=100, height=112, nChannels=3, hidden_size=100, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)
pretrained_model = args.modelname
vae.loadweight_from(pretrained_model)
# print(vae)
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)
vae.test(test_loader, save_folder=args.save_folder)

