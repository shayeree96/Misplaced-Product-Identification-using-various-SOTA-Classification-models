import os
import random
from skimage import io
import imageio
from random import randint
from PIL import Image
import torch
import torch.utils.data
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convPairVAE import ConvPairVAE
import pandas as pd
#from dataset import dataset
import argparse
from udlp.utils import RandomMultipleCrop
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--nFilters', type=int, default=256)
parser.add_argument('--nFeature', type=int, default=32)
parser.add_argument('--modelname', type=str, default='trainB3_12')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.1, help='decreasing learning rate by gamma (default: 0.1)')
parser.add_argument('--step-value', type=int, default=[10, 20, 30], nargs='+')
args = parser.parse_args()
print(args)

assert torch.cuda.is_available(), 'Error: CUDA not found!'

rotation = transforms.Compose([transforms.RandomRotation(10), transforms.ToTensor()])
h_crop = transforms.Compose([transforms.Resize((153+5, 116)), transforms.RandomCrop((153, 116)),
                                transforms.ToTensor()])
w_crop = transforms.Compose([transforms.Resize((153, 116+5)), transforms.RandomCrop((153, 116)),
                            transforms.ToTensor()])
crop = transforms.Compose([transforms.Resize((153+15, 116+15)), transforms.RandomCrop((153, 116)),
                            transforms.ToTensor()])
normal = transforms.Compose([transforms.ToTensor()])

class PairDataset():


    def __init__(self, txt_file, transform):
        
        self.img_pair = pd.read_csv(txt_file, sep=",", engine='python')
        self.img_pair.dropna(inplace=True)
        self.transform = transform
        

    def __len__(self):
        return len(self.img_pair)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_x = self.img_pair.iloc[idx, 0]  
        img_y = self.img_pair.iloc[idx, 1]
        im1 = plt.imread(img_x)
        im2 = plt.imread(img_y)
        f=im1.shape
        
        if len(f)<=2:
            im_new=im1.reshape((im1.shape[0], im1.shape[1], 1))
            im1=np.concatenate((im_new,im_new),axis=2)
            im1=np.concatenate((im1,im_new),axis=2)
  
        im1 = Image.fromarray(im1, mode='RGB')
        im2 = Image.fromarray(im2, mode='RGB')

        label=int(self.img_pair.iloc[idx, 2])

        ind1 = randint(0, 2)
        ind2 = randint(0, 2)
        
        if self.transform is not None:

            if len(self.transform) > 1:

                im1 = self.transform[ind1](im1)
                im2 = self.transform[ind2](im2)

            else:
                im1 = self.transform[0](im1 )
                im2 = self.transform[0](im2 )

        return im1,im2,label

train_loader = torch.utils.data.DataLoader(
    PairDataset('/home/ironman/shayeree/vae/train_try_4cls.txt',
                transform=[h_crop,w_crop,crop]),batch_size=args.batch_size, shuffle=False, num_workers=44)

valid_loader = torch.utils.data.DataLoader(
PairDataset('/home/ironman/shayeree/vae/valid_try_4cls.txt', transform=[h_crop,w_crop,crop]),
    batch_size=args.batch_size, shuffle=False, num_workers=44)

test_loader = torch.utils.data.DataLoader(
PairDataset('/home/ironman/shayeree/vae/test_try_4cls.txt',transform=[h_crop,w_crop,crop]),
    batch_size=args.batch_size, shuffle=False, num_workers=44)

vae = ConvPairVAE(width=116, height=153, nChannels=3, hidden_size=100, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)

pretrained_model = '/home/ironman/shayeree/vae/models/trainB3_12256_32_epoch-10'#Autoencoder path here
vae.loadweight_from(pretrained_model)

print(vae)
vae.cuda()
#vae=torch.nn.DataParallel(vae.cuda(), device_ids=range(torch.cuda.device_count()))
vae.pair_fit(train_loader, valid_loader, lr=args.lr, num_epochs=args.epochs, step_value=args.step_value, gamma=args.gamma)

