import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
from udlp.attn import Self_Attn
import os
import numpy as np
from udlp.vgg import vgg16_bn


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

    net.append(nn.Conv2d(4 * nFilters, hidden_size, kernel_size=4))
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


class VGGBranchVAE(nn.Module):
    def __init__(self, width=32, height=32, nChannels=3, hidden_size=500, z_dim=20, binary=True,
                 nFilters=64, n_class=2, name='vae'):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.width = width
        self.height = height
        self.nChannels = nChannels
        self.name = '{}{}_{}'.format(name, nFilters, z_dim)
        if not os.path.exists('results/' + self.name):
            os.makedirs('results/' + self.name)
        if not os.path.exists('results/' + self.name + '/reconstruct'):
            os.makedirs('results/' + self.name + '/reconstruct')
        self.encoder_z = vgg16_bn()
        self.encoder_w = vgg16_bn()
        self.decoder = buildDecoderNetwork(hidden_size*2, nFilters, nChannels)

        self._enc_mu_w = nn.Linear(hidden_size * 3 * 2, z_dim)
        self._enc_log_sigma_w = nn.Linear(hidden_size * 3 * 2, z_dim)
        # self._enc_mu_w = nn.Conv2d(hidden_size, z_dim, kernel_size=1)
        # self._enc_log_sigma_w = nn.Conv2d(hidden_size, z_dim, kernel_size=1)

        self._enc_mu_z = nn.Linear(hidden_size * 3 * 2, z_dim)
        self._enc_log_sigma_z = nn.Linear(hidden_size * 3 * 2, z_dim)

        self._dec_w = nn.Sequential(nn.Linear(z_dim, hidden_size * 11 * 7),
                                    nn.BatchNorm1d(hidden_size * 11 * 7), nn.ReLU(True))
        # self._dec_w = nn.Sequential(nn.ConvTranspose2d(z_dim, hidden_size, kernel_size=1),
        #                             nn.BatchNorm2d(hidden_size), nn.ReLU(True))
        self._dec_z = nn.Sequential(nn.Linear(z_dim, hidden_size * 11 * 7),
                                    nn.BatchNorm1d(hidden_size * 11 * 7), nn.ReLU(True))
        self._dec_act = None
        self.enc_attn_z = Self_Attn(hidden_size)
        self.enc_attn_w = Self_Attn(hidden_size)
        self.dec_attn_z = Self_Attn(hidden_size)
        self.dec_attn_w = Self_Attn(hidden_size)
        self.metrics = nn.Sequential(
            nn.Linear(z_dim*2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=True),
            # nn.ReLU(inplace=True)
        )
        # self.metrics = nn.Linear(z_dim, 512, bias=False)

        self.cls_fc = nn.Linear(512, n_class, bias=False)

        if binary:
            self._dec_act = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, w, z):
        h_w = self._dec_w(w).view(w.size(0), self.hidden_size, 11, 7)
        h_z = self._dec_z(z).view(z.size(0), self.hidden_size, 11, 7)
        h = torch.cat((self.dec_attn_w(h_w)[0], self.dec_attn_z(h_z)[0]), 1)
        # h = self.dec_attn_z(h_z)[0]
        # h = self.dec_attn_w(h_w)[0]
        x = self.decoder(h)
        # x = F.upsample(x, (self.height, self.width), mode='bilinear')
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def decode_w(self, w):
        h_w = self._dec_w(w).view(w.size(0), self.hidden_size, 11, 7)
        h = self.dec_attn_w(h_w)[0]
        x = self.decoder(torch.cat((h, h * 0), 1))
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def decode_z(self, z):
        h_z = self._dec_z(z).view(z.size(0), self.hidden_size, 11, 7)
        h = self.dec_attn_z(h_z)[0]
        x = self.decoder(torch.cat((h * 0, h), 1))
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x, mu_w, logvar_w, logvar_z, target_var=10e-4):
        recon_x = recon_x.view(recon_x.size(0), -1)
        x = x.view(x.size(0), -1)

        BCE = -(x * torch.log(recon_x.clamp(min=1e-10)) +
                (1 - x) * torch.log((1 - recon_x).clamp(min=1e-10))).sum(1)
        KLD_w = -0.5 * (1 + logvar_w - mu_w.pow(2) - logvar_w.exp())\
            .view(mu_w.shape[0], mu_w.shape[1], -1).mean(2).sum(1)
        KLD_z = -0.5 * (1 + logvar_z - logvar_z.exp() / target_var)\
            .view(logvar_z.shape[0], logvar_z.shape[1], -1).mean(2).sum(1)
        # Normalise by same number of elements as in reconstruction
        loss = torch.mean(BCE + KLD_w + KLD_z)

        return loss

    def pair_loss(self, feat, y):
        feat = self.metrics(feat)
        loss = F.nll_loss(F.log_softmax(self.cls_fc(feat), dim=1), y)
        return loss

    def cls_loss(self, feat, y):
        feat = self.metrics(feat)
        loss = F.nll_loss(F.log_softmax(self.cls_fc(feat), dim=1), y)
        return loss

    def forward(self, x):
        h_z, h_w = self.encoder_z(x), self.encoder_w(x)
        h_z = self.enc_attn_z(h_z)[0]
        h_w = self.enc_attn_w(h_w)[0]
        mu_w = self._enc_mu_w(h_w.view(h_z.size(0), -1))
        mu_z = self._enc_mu_z(h_z.view(h_z.size(0), -1))
        logvar_w = self._enc_log_sigma_w(h_w.view(h_z.size(0), -1))
        logvar_z = self._enc_log_sigma_z(h_z.view(h_z.size(0), -1))
        w = self.reparameterize(mu_w, logvar_w)
        z = self.reparameterize(mu_z, logvar_z)
        return self.decode(w, z), mu_w, mu_z, logvar_w, logvar_z

    def abstract_feature(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return z

    def loadweight_from(self, pretrain_path):
        pretrained_dict = torch.load(pretrain_path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def fit(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, log_interval=100):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            torch.set_grad_enabled(True)
            train_loss = 0
            train_cls_loss = 0
            for batch_idx, (inputs, y) in enumerate(trainloader):
                # if batch_idx > 10:
                #     break
                if len(inputs.shape) > 4:
                    shp = inputs.shape
                    inputs = inputs.view(-1, shp[2], shp[3], shp[4])
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)

                outputs, mu_w, mu_z, logvar_w, logvar_z = self.forward(inputs)
                recon_loss = self.loss_function(outputs, inputs, mu_w, logvar_w, logvar_z)
                cls_loss = self.cls_loss(mu_z, y)
                loss = recon_loss + cls_loss
                print(batch_idx)
                loss.backward()
                train_loss += recon_loss.item() * len(inputs)
                train_cls_loss += cls_loss.item() * len(inputs)
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        "#epoch %3d: train loss: %.5f | train cls loss: %.5f" % (
                            epoch, recon_loss.item(), cls_loss.item()))
                    sample = self.decode_w(mu_w).cpu()
                    save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                               'results/' + self.name + '/sample_w' + str(epoch) + '-' + str(batch_idx) + '.png')
                    sample = self.decode_z(mu_z).cpu()
                    save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                               'results/' + self.name + '/sample_z' + str(epoch) + '-' + str(batch_idx) + '.png')
                    sample = self.decode(mu_w, mu_z).cpu()
                    save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                               'results/' + self.name + '/sample_' + str(epoch) + '-' + str(batch_idx) + '.png')

            # validate
            self.eval()
            torch.set_grad_enabled(False)
            valid_loss = 0.0
            valid_cls_loss = 0.0
            total = 0
            correct = 0
            for batch_idx, (inputs, y) in enumerate(validloader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                    y = y.cuda()
                inputs = Variable(inputs)
                outputs, mu_w, mu_z, logvar_w, logvar_z = self.forward(inputs)

                loss = self.loss_function(outputs, inputs, mu_w, logvar_w, logvar_z)
                valid_loss += loss.item() * len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]

                # view reconstruct
                if batch_idx % log_interval == 0:
                    n = min(inputs.size(0), 8)
                    comparison = torch.cat([inputs.view(-1, self.nChannels, self.height, self.width)[:n],
                                            outputs.view(-1, self.nChannels, self.height, self.width)[:n]])
                    save_image(comparison.data.cpu(),
                               os.path.join('results', self.name, 'reconstruct', '{}.png'.format(epoch)), nrow=n)
                    # 'results/' + self.name + '/reconstruct/reconstruction_' + str(epoch) + '.png', nrow=n)
                    # print(
                    #     strftime("%Y-%m-%d %H:%M:%S", localtime())
                    #     + '\tTrain Epoch: {} [{}/{}]\t'.format(epoch, batch_idx * len(trainloader),
                    #                                            len(self.trainloader))
                    #     + '\tTrain loss: {}, Valid Loss: {}'.format(train_loss / len(trainloader.dataset),
                    #                                                 valid_loss / len(validloader.dataset)))
                cls_loss = self.cls_loss(mu_z, y)
                valid_cls_loss += cls_loss.item() * len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]
                total += y.size(0)
                _, predicted = self.cls_fc(self.metrics(mu_z)).max(1)
                correct += predicted.eq(y).sum().item()

            valid_acc = correct / total

            # valid_loss = total_loss / total_num
            print("#epoch %3d: train loss: %.5f | train cls loss: %.5f | valid loss: %.5f | valid cls loss: %.5f"
                  " | valid acc: %.4f" % (
                      epoch, train_loss / len(trainloader.dataset) / 2, train_cls_loss / len(trainloader.dataset),
                      valid_loss / len(validloader.dataset) / 2, valid_cls_loss / len(validloader.dataset), valid_acc))
            # valid_loss = total_loss / total_num

            # print("#Epoch %3d: Train Loss: %.5f, Valid Loss: %.5f" % (
            #     epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

            sample = Variable(torch.randn(64, self.z_dim, 1, 1))
            if use_cuda:
                sample = sample.cuda()
            # sample = self.decode(sample).cpu()
            # save_image(sample.data.view(64, self.nChannels, self.height, self.width),
            #            'results/' + self.name + '/sample_' + str(epoch) + '.png')


            sample = self.decode_w(mu_w).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_w' + str(epoch) + '.png')
            sample = self.decode_z(mu_z).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_z' + str(epoch) + '.png')
            sample = self.decode(mu_w, mu_z).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_' + str(epoch) + '.png')

    def fit_pair(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, log_interval=100):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            torch.set_grad_enabled(True)
            train_loss = 0
            train_cls_loss = 0
            for batch_idx, (pair1, pair2, y) in enumerate(trainloader):
                if len(pair1.shape) > 4:
                    shp = pair1.shape
                    pair1 = pair1.view(-1, shp[2], shp[3], shp[4])
                    pair2 = pair2.view(-1, shp[2], shp[3], shp[4])
                pair1, pair2 = pair1.float(), pair2.float()
                if use_cuda:
                    pair1, pair2 = pair1.cuda(), pair2.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                pair1, pair2 = Variable(pair1), Variable(pair2)

                outputs1, mu_w1, mu_z1, logvar_w1, logvar_z1 = self.forward(pair1)
                outputs2, mu_w2, mu_z2, logvar_w2, logvar_z2 = self.forward(pair2)

                recon_loss1 = self.loss_function(outputs1, pair1, mu_w1, logvar_w1, logvar_z1)
                recon_loss2 = self.loss_function(outputs2, pair2, mu_w2, logvar_w2, logvar_z2)
                recon_loss = recon_loss1 + recon_loss2
                feat1 = mu_z1.squeeze()
                feat2 = mu_z2.squeeze()

                cls_loss = self.cls_loss(torch.cat((feat1, feat2), 1), y)
                loss = recon_loss + cls_loss
                train_loss += recon_loss.data * len(pair1) * 2
                train_cls_loss += cls_loss.data * len(pair1)
                loss.backward()
                optimizer.step()
                # print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                #     batch_idx, recon_loss.data[0]))

            # validate
            self.eval()
            torch.set_grad_enabled(False)
            valid_loss = 0.0
            valid_cls_loss = 0.0
            total = 0
            correct = 0
            tp = 0
            tn = 0
            pos = 0
            neg = 0
            for batch_idx, (pair1, pair2, y) in enumerate(validloader):
                pair1, pair2 = pair1.float(), pair2.float()
                if use_cuda:
                    pair1, pair2 = pair1.cuda(), pair2.cuda()
                    y = y.cuda()
                pair1, pair2 = Variable(pair1), Variable(pair2)

                outputs1, mu_w1, mu_z1, logvar_w1, logvar_z1 = self.forward(pair1)
                outputs2, mu_w2, mu_z2, logvar_w2, logvar_z2 = self.forward(pair2)

                recon_loss1 = self.loss_function(outputs1, pair1, mu_w1, logvar_w1, logvar_z1)
                recon_loss2 = self.loss_function(outputs2, pair2, mu_w2, logvar_w2, logvar_z2)

                recon_loss = recon_loss1 + recon_loss2
                feat1 = mu_z1.squeeze()
                feat2 = mu_z2.squeeze()
                cls_loss = self.cls_loss(torch.cat((feat1, feat2), 1), y)
                loss = recon_loss + cls_loss

                valid_loss += loss.data * len(pair1) * 2
                valid_cls_loss += cls_loss * len(pair1)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]
                total += y.size(0)
                pos += y.sum()
                neg += (1-y).sum()
                _, predicted = self.cls_fc(self.metrics(torch.cat((feat1, feat2), 1))).max(1)
                correct += predicted.eq(y).sum().item()
                tp += (predicted * y).sum().item()
                tn += ((1-predicted) * (1-y)).sum().item()

                # view reconstruct
                if batch_idx % log_interval == 0:
                    n = min(pair1.size(0), 8)
                    comparison = torch.cat([pair1.view(-1, self.nChannels, self.height, self.width)[:n],
                                            outputs1.view(-1, self.nChannels, self.height, self.width)[:n]])
                    save_image(comparison.data.cpu(),
                               os.path.join('results', self.name, 'reconstruct', '{}.png'.format(epoch)), nrow=n)

            valid_acc = correct / total
            tpr = float(tp) / float(pos)
            tnr = float(tn) / float(neg)
            # valid_loss = total_loss / total_num
            print("#epoch %3d: train loss: %.5f | train cls loss: %.5f | valid loss: %.5f | valid cls loss: %.5f"
                  " | valid acc: %.4f | tpr: %.4f | tnr: %.4f" % (
                epoch, train_loss / len(trainloader.dataset) / 2, train_cls_loss / len(trainloader.dataset),
                valid_loss / len(validloader.dataset) / 2, valid_cls_loss / len(validloader.dataset), valid_acc, tpr, tnr))

            sample_w = Variable(torch.randn(64, self.z_dim, 2, 3))
            sample_z = Variable(torch.randn(64, self.z_dim, 2, 3))
            if use_cuda:
                sample_w, sample_z = sample_w.cuda(), sample_z.cuda()
            # sample = self.decode(sample_w, sample_z).cpu()
            sample = self.decode_w(mu_w1).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_w' + str(epoch) + '.png')
            sample = self.decode_z(mu_z1).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_z' + str(epoch) + '.png')
            sample = self.decode(mu_w1, mu_z1).cpu()
            save_image(sample.data.view(-1, self.nChannels, self.height, self.width),
                       'results/' + self.name + '/sample_' + str(epoch) + '.png')

            if (epoch + 1) % 10 == 0:
                torch.save(self.state_dict(), 'models/{}_epoch-{}'.format(self.name, epoch))

    def test(self, testloader):
        self.eval()
        torch.set_grad_enabled(False)
        if not os.path.isdir('results/' + self.name + '/reconstruct/test'):
            os.makedirs('results/' + self.name + '/reconstruct/test')
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs, mu, logvar = self.forward(inputs)

            # view reconstruct
            n = min(inputs.size(0), 8)
            for i in range(0, inputs.shape[0], n):
                comparison = torch.cat([inputs.view(-1, self.nChannels, self.height, self.width)[i:i + n],
                                        outputs.view(-1, self.nChannels, self.height, self.width)[i:i + n]])
                save_image(comparison.data.cpu(),
                           'results/' + self.name + '/reconstruct/test/rec_group_{}_{}'.format(batch_idx, i) + '.png',
                           nrow=n)

    def test_feature(self, testloader):
        self.eval()
        torch.set_grad_enabled(False)
        use_cuda = torch.cuda.is_available()
        features = []
        if use_cuda:
            self.cuda()
        for batch_idx, (inputs, _) in enumerate(testloader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            feat = self.abstract_feature(inputs)
            features.append(feat.cpu())
        return torch.cat(features)
