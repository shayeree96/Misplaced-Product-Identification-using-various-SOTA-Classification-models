import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from time import strftime, localtime
import numpy as np
import math

# from udlp.utils import Dataset, masking_noise
# from udlp.ops import MSELoss, BCELoss

def buildEncoderNetwork(input_channels, nFilters, hidden_size):
    net = []
    net.append(nn.Conv2d(input_channels, nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(nFilters, 2 * nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(2 * nFilters, 4 * nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(4 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.Conv2d(4 * nFilters, hidden_size, kernel_size=3))
    net.append(nn.BatchNorm2d(hidden_size))
    net.append(nn.ReLU(True))

    return nn.Sequential(*net)


def buildDecoderNetwork(hidden_size, nFilters, output_channels):
    net = []
    net.append(nn.ConvTranspose2d(hidden_size, 4 * nFilters, kernel_size=4))
    net.append(nn.BatchNorm2d(4 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(4 * nFilters, 2 * nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(2 * nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(2 * nFilters, nFilters, kernel_size=4, stride=2, padding=1))
    net.append(nn.BatchNorm2d(nFilters))
    net.append(nn.ReLU(True))

    net.append(nn.ConvTranspose2d(nFilters, output_channels, kernel_size=4, stride=2, padding=1))
    return nn.Sequential(*net)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, c_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, c_size, kernel_size=3, stride=1, padding=1)
        self.conv1_relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_size, c_size, kernel_size=3, stride=1, padding=1)
        self.conv2_relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv1_relu(out)
        out = self.conv2(out)
        out = self.conv2_relu(out)
        out += residual
        return out


class ConvPairVAE(nn.Module):
    def __init__(self, width=32, height=32, nChannels=3, hidden_size=100, z_dim=20, binary=True,
                 nFilters=64, name='vae'):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.nChannels = nChannels
        self.name = '{}{}_{}'.format(name, nFilters, z_dim)

        self.encoder = buildEncoderNetwork(nChannels, nFilters, hidden_size)
        self.decoder = buildDecoderNetwork(hidden_size, nFilters, nChannels)
        self._enc_mu = nn.Conv2d(hidden_size, z_dim, kernel_size=1)
        self._enc_log_sigma = nn.Conv2d(hidden_size, z_dim, kernel_size=1)
        self._dec = nn.ConvTranspose2d(z_dim, hidden_size, kernel_size=1)
        self._dec_bn = nn.BatchNorm2d(hidden_size)
        self._dec_relu = nn.ReLU(True)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()
        self.cls_head = self._make_head(z_dim, 1, 256)
        self.cls_fc = nn.Linear(17*12, 2)

    def _make_head(self, in_planes, out_planes, c_size):
        layers = []
        layers.append(nn.Conv2d(in_planes * 2, c_size, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(True))
        for _ in range(3):
            layers.append(nn.Conv2d(c_size, c_size, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(c_size, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    # def _make_head(self, block, in_planes, out_planes, c_size):
    #     layers = []
    #     layers.append(block(in_planes*2, c_size))
    #     layers.append(block(c_size, c_size))
    #     layers.append(nn.Conv2d(c_size, out_planes, kernel_size=3, stride=1, padding=1))
    #     return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = self._dec_relu(self._dec_bn(self._dec(z)))
        x = self.decoder(h)
        x = F.upsample(x, (self.height, self.width), mode='bilinear')
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, mu, logvar):
        recon_x = recon_x.view(recon_x.size(0), -1)
        x = x.view(x.size(0), -1)
        BCE = -torch.sum(x * torch.log(torch.clamp(recon_x, min=1e-10)) +
                         (1 - x) * torch.log(torch.clamp(1 - recon_x, min=1e-10)), 1)
        KLD = -0.5 * torch.sum((1 + logvar - mu.pow(2) - logvar.exp()).view(mu.shape[0], mu.shape[1], -1).mean(2), 1)
        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + KLD)
        return loss

    def cls_loss(self, x, y): 
        loss = F.nll_loss(F.log_softmax(x, dim=1), y)
        return loss

    def forward(self, x):
        h = self.encoder(x) 
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def pair_forward(self, x1, x2):
        f = torch.cat((x1, x2), dim=1)
        #print('Dimensions of x1 :{} and x2:{}',x1.shape,x2.shape)
        #print('Shape of x1 and x2 concatenation :',f.shape)
        patch_cls = self.cls_head(f)
        #print('shape pf patch_cls :',patch_cls.shape)
        
        patch_cls = patch_cls.view(-1,17*12)
        #print('shape pf patch_cls :',patch_cls.shape)
        y = self.cls_fc(patch_cls)
        #print('shape of return in pair forward',y.shape)
        return y

    def loadweight_from(self, pretrain_path):
        pretrained_dict = torch.load(pretrain_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def pair_fit(self, trainloader, validloader, lr=0.001, num_epochs=10, step_value=(10, 20), gamma=0.1):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        device_ids=range(torch.cuda.device_count())
        
        #self=torch.nn.DataParallel(self.cuda(), device_ids=range(torch.cuda.device_count()))

        for epoch in range(num_epochs):
            if epoch in step_value:
                lr = lr * gamma
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
            # train 1 epoch
            self.train()
            torch.set_grad_enabled(True)
            train_loss = 0.0
            for batch_idx, (input1, input2, label) in enumerate(trainloader):
                print('E: %d, B: %d / %d' % (epoch+1, batch_idx+1, len(trainloader)), end=' |\n ')
                input1 = input1.float()
                input2 = input2.float()
                if use_cuda:
                    input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
                optimizer.zero_grad()
                input1, input2, label = Variable(input1), Variable(input2), Variable(label)
                rec_x1, mu1, logvar1 = self.forward(input1)
                rec_x2, mu2, logvar2 = self.forward(input2)
                
                feat = self.pair_forward(mu1, mu2)
                loss = self.cls_loss(feat, label)

                #print('shape of y label :',label.shape)
                loss.backward()
                optimizer.step()
                train_loss += loss

            # validate
            self.eval()
            torch.set_grad_enabled(False)
            valid_loss = 0.0
            total = 0
            correct = 0
            tp = 0
            tn = 0
            pos = 0
            neg = 0
            
            ckpt_dir = os.path.join('./models/pair_cls')
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            for batch_idx, (input1, input2, label) in enumerate(validloader):
                print('Eval E: %d, B: %d / %d' % (epoch+1, batch_idx+1, len(validloader)), end=' |\n ')
                input1 = input1.float()
                input2 = input2.float()
                if use_cuda:
                    input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
                input1, input2, label = Variable(input1), Variable(input2), Variable(label)
                rec_x1, mu1, logvar1 = self.forward(input1)
                rec_x2, mu2, logvar2 = self.forward(input2)
                feat = self.pair_forward(mu1, mu2)
                #print('shape pf mul1 and mul2',mu1.shape,mu2.shape)
                valid_loss += self.cls_loss(feat, label)
                # compute the accuracy
                total += label.size(0)
                _, predicted = torch.max(feat.data, 1)
                pos += label.sum()
                neg += (1 - label).sum()
                correct += (predicted == label).sum().item()
                tp += (predicted * label).sum().item()
                tn += ((1 - predicted) * (1 - label)).sum().item()
            valid_acc = correct / total
            tpr = float(tp) / float(pos)
            tnr = float(tn) / float(neg)
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
            print("#Epoch {}: Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f}".
                   format(epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
            #torch.save(self.state_dict(), 'models/pair_cls/{}_{:.4f}_epoch-{}'.format(self.name, valid_acc, epoch))
            torch.save(self.state_dict(), os.path.join(ckpt_dir,'ckpt_{}_4cls_ valid acc: {:.4f}_valid_loss: {:.4f} epoch-{}.pth'.format(self.name, valid_acc, (valid_loss/len(validloader.dataset)),epoch+1)))

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            torch.set_grad_enabled(True)
            train_recons_loss = 0.0
            for batch_idx, (input1, input2, label) in enumerate(trainloader):
                input1 = input1.float()
                input2 = input2.float()
                if use_cuda:
                    input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
                    
                optimizer.zero_grad()
                input1, input2, label = Variable(input1), Variable(input2), Variable(label)
                rec_x1, mu1, logvar1 = self.forward(input1)
                rec_x2, mu2, logvar2 = self.forward(input2)
                feat = self.pair_forward(mu1, mu2)
                cls_loss = self.cls_loss(feat, label)
                recons_loss = self.loss_function(rec_x1, input1, mu1, logvar1) + self.loss_function(rec_x2, input2, mu2, logvar2)
                loss = recons_loss + cls_loss
                loss.backward()
                optimizer.step()
                train_recons_loss += recons_loss

            # validate
            self.eval()
            torch.set_grad_enabled(False)
            valid_loss = 0.0
            total = 0
            correct = 0
            tp = 0
            tn = 0
            pos = 0
            neg = 0
            for batch_idx, (input1, input2, label) in enumerate(validloader):
                input1 = input1.float()
                input2 = input2.float()
                if use_cuda:
                    input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
                input1, input2, label = Variable(input1), Variable(input2), Variable(label)
                rec_x1, mu1, logvar1 = self.forward(input1)
                rec_x2, mu2, logvar2 = self.forward(input2)
                feat = self.pair_forward(mu1, mu2)
                valid_loss += self.cls_loss(feat, label)
                # compute the accuracy
                total += label.size(0)
                pos += label.sum()
                neg += (1 - label).sum()
                _, predicted = torch.max(self.cls_fc(feat).data, 1)
                correct += (predicted == label).sum().item()
                tp += (predicted * label).sum().item()
                tn += ((1 - predicted) * (1 - label)).sum().item()
            valid_acc = correct / total

            print(strftime("%Y-%m-%d %H:%M:%S", localtime()) + '\tTrain Epoch: {} [{}/{}]\t'.
                  format(epoch, batch_idx * len(trainloader), len(self.trainloader)))
            print ("#Epoch {}: Train Loss: {}, Valid Loss: {}, Valid Accuracy: {}, tpr: {}, tnr: {}".
                   format(epoch, train_recons_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset),
                          valid_acc, tp / pos, tn / neg))

    def test(self, testloader):
        self.eval()
        torch.set_grad_enabled(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        valid_loss = 0.0
        total = 0
        correct = 0
        predict = []
        k=[]
        for batch_idx, (input1, input2, label, img_name) in enumerate(testloader):
            print(' B: %d / %d' % (batch_idx+1, len(testloader)), end=' |\n ')
            input1 = input1.float()
            input2 = input2.float()
            if use_cuda:
                    input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
            input1, input2, label = Variable(input1), Variable(input2), Variable(label)
        
            rec_x1, mu1, logvar1 = self.forward(input1)
            rec_x2, mu2, logvar2 = self.forward(input2)
            feat = self.pair_forward(mu1, mu2)
            #print('mu 1 and mu 2 shape:',mu1.shape,mu2.shape)
            valid_loss += self.cls_loss(feat, label)
            # compute the accuracy
            total += label.size(0)
           # k=self.cls_fc(feat)
           # print("k shape :",k.shape)

            _, predicted = torch.max(feat, 1)
            wrong_pred=np.where(torch.Tensor.cpu(predicted).detach().numpy()!=torch.Tensor.cpu(label).detach().numpy())
            wrong_pred=np.transpose(wrong_pred)
            wrong_pred=wrong_pred.flatten()
            
            for i in wrong_pred:
                k.append(img_name[int(i)])
            #print(k)
            #print("No of wrong predictions :",wrong_pred)
            #print("Type for slicing or chooping :",np.shape(wrong_pred))
            correct += (predicted == label).sum().item()
            #print('shape of wrongs :',k.shape)
            #print("Image nam : ",img_name[_])
            predict.append(predicted.cpu())
        valid_acc = correct / total

        print ("#Valid Loss: {}, Valid Accuracy: {}".
               format(valid_loss / len(testloader.dataset), valid_acc))
        #torch.save(self.state_dict(), os.path.join(ckpt_dir,'ckpt_{}_{:.4f}_epoch-{}.pth'.format(self.name, valid_acc, epoch)))       
        return torch.cat(predict), valid_acc,k