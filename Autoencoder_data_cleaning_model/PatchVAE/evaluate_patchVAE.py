import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
from udlp.autoencoder.convPairVAE import ConvPairVAE
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='VAE PairWise Data Example')
parser.add_argument('--lr', type=float, default=0.0001, metavar='N',
                    help='learning rate for training (default: 0.001)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--nFilters', type=int, default=256)
parser.add_argument('--nFeature', type=int, default=32)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--modelname', type=str, default='trainB3_12')
args = parser.parse_args()
save_folder='/home/ironman/shayeree/PatchVAE/evaluation_results'

class PairDataset():

    def __init__(self, txt_file, transform):
        
        self.img_pair = pd.read_csv(txt_file, sep=",", engine='python')
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
        
        if self.transform is not None:
            im1 = self.transform(im1)
            im2 = self.transform(im2)

        return im1,im2,label,img_y
        
test_loader = torch.utils.data.DataLoader(
PairDataset('/home/ironman/shayeree/vae/test_try_new.txt',transforms.Compose([transforms.Resize((153+5, 116)), transforms.RandomCrop((153, 116)),
                                transforms.ToTensor()])),
    batch_size=args.batch_size, shuffle=False, num_workers=44)

vae = ConvPairVAE(width=256, height=128, nChannels=3, hidden_size=100, z_dim=args.nFeature, binary=True,
        nFilters=args.nFilters, name=args.modelname)
pretrained_model = '/home/ironman/shayeree/PatchVAE/models/pair_cls/ckpt_trainB3_12256_32_ new_ valid acc: 0.8446_valid_loss: 0.0076 epoch-16.pth'
vae.loadweight_from(pretrained_model)
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
predict,valid_acc,k=vae.test(test_loader)

print("THe accuracy is  :",valid_acc)

with open('miss_predictions.txt', 'w') as filehandle:
    for listitem in k:
        
        filehandle.write('%s\n' % listitem)
