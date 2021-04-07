import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convPairVAE import ConvPairVAE
from dataset import dataset
import argparse
from udlp.utils import RandomMultipleCrop
# from dataset.data_loader import PanoDataset

from dataset.dataset import PairDataset
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nFilters', type=int, default=128)
parser.add_argument('--nFeature', type=int, default=250)
parser.add_argument('--modelname', type=str, default='vae_')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--bb-train', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.1, help='decreasing learning rate by gamma (default: 0.1)')
parser.add_argument('--step-value', type=int, default=[10, 20, 30], nargs='+')
args = parser.parse_args()
print(args)

if args.bb_train:
    folder = '/media/ran/Suibian/Project/autoencoder/dataset/unalligned_new_data_w0_h0/'
    normal = transforms.Compose([transforms.Resize((153, 116)), transforms.ToTensor()])
    rotation = transforms.Compose([transforms.RandomRotation(10), transforms.Resize((153, 116)), transforms.ToTensor()])
    h_crop = transforms.Compose([transforms.Resize((153 + 5, 116)), transforms.RandomCrop((153, 116)),
                                 transforms.ToTensor()])
    w_crop = transforms.Compose([transforms.Resize((153, 116 + 5)), transforms.RandomCrop((153, 116)),
                                 transforms.ToTensor()])
    crop = transforms.Compose([transforms.Resize((153 + 15, 116 + 15)), transforms.RandomCrop((153, 116)),
                               transforms.ToTensor()])
    # normal = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/size_checked_pair_trainset.txt', '',
                    transform=[rotation, h_crop, w_crop, crop, normal]),
        batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/size_checked_pair_validset.txt', '',
                    transform=[normal]),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/size_checked_pair_testset.txt', '',
                    transform=[normal]),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

else:
    folder = '/media/ran/Suibian/Project/autoencoder/dataset/new_data_w116_h153/'
    rotation = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()])
    h_crop = transforms.Compose([transforms.Resize((153+5, 116)), transforms.RandomCrop((153, 116)),
                                 transforms.ToTensor()])
    w_crop = transforms.Compose([transforms.Resize((153, 116+5)), transforms.RandomCrop((153, 116)),
                                transforms.ToTensor()])
    crop = transforms.Compose([transforms.Resize((153+15, 116+15)), transforms.RandomCrop((153, 116)),
                                transforms.ToTensor()])
    normal = transforms.Compose([transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/pairwise_trainset.txt', '',
                    transform=[rotation,h_crop,w_crop,crop,normal]),
        batch_size=args.batch_size, shuffle=True, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/pairwise_validset.txt', '',
                    transform=[normal]),
        batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        PairDataset(folder, 'dataset/new_pair_txt/pairwise_testset.txt', '',
                    transform=[normal]),
        batch_size=args.batch_size, shuffle=False, num_workers=2)

vae = ConvPairVAE(width=116, height=153, nChannels=3, hidden_size=500, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)

pretrained_model = 'models/best_model49.pth'#Autoencoder path here
vae.loadweight_from(pretrained_model)
print(vae)
vae.pair_fit(train_loader, valid_loader, lr=args.lr, num_epochs=args.epochs, step_value=args.step_value, gamma=args.gamma)

