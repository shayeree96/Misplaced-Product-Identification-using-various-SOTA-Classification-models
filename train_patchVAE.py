import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from convPatchVAE import ConvPatchVAE
import dataset
import argparse


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nFilters', type=int, default=256)
parser.add_argument('--nFeature', type=int, default=32)
parser.add_argument('--modelname', type=str, default='trainB3_12')
parser.add_argument('--pretrained', type=bool, default=False)
args = parser.parse_args()


train_loader = torch.utils.data.DataLoader(
    dataset.Digits('trainlist.txt',
                   # transform=transforms.Compose([transforms.Resize((64, 118)), transforms.ToTensor()])),
                   transform=transforms.Compose([transforms.Resize((112, 100)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    dataset.Digits('testlist.txt',
                   # transform=transforms.Compose([transforms.Resize((64, 118)), transforms.ToTensor()])),
                   transform=transforms.Compose([transforms.Resize((112, 100)), transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False, num_workers=2)

vae = ConvPatchVAE(width=100, height=112, nChannels=3, hidden_size=100, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)

print(vae)
vae.fit(train_loader, test_loader, lr=args.lr, num_epochs=args.epochs)
torch.save(vae.state_dict(), 'models/{}_epoch-{}'.format(vae.name, args.epochs))
