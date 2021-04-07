import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convPatchVAE import ConvPatchVAE
from dataset import dataset
import argparse
from udlp.utils import RandomMultipleCrop

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nFilters', type=int, default=128)
parser.add_argument('--nFeature', type=int, default=250)
parser.add_argument('--modelname', type=str, default='trainB3_12')
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()


train_loader = torch.utils.data.DataLoader(
    dataset.Digits('dataset/train_exp.txt', folder=['B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12'],
                   data_folder='VAE_data',
                   # transform=transforms.Compose([transforms.Resize((64, 118)), transforms.ToTensor()])),
                   transform=transforms.Compose([transforms.Resize((32, 24)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset.Digits('dataset/test_exp.txt', folder=['B12'], data_folder='VAE_data',
                   # transform=transforms.Compose([transforms.Resize((64, 118)), transforms.ToTensor()])),
                   transform=transforms.Compose([transforms.Resize((32, 24)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False, num_workers=2)

vae = ConvPatchVAE(width=24, height=32, nChannels=3, hidden_size=100, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)

print(vae)
vae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs)
torch.save(vae.state_dict(), 'models/{}_epoch-{}'.format(vae.name, args.epochs))
