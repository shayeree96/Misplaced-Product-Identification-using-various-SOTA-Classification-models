import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.utils import save_image
import os


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


class ConvPatchVAE(nn.Module):
    def __init__(self, width=32, height=32, nChannels=3, hidden_size=500, z_dim=20, binary=True,
                 nFilters=64, name='vae'):
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

    def forward(self, x):
        h = self.encoder(x)
        mu = self._enc_mu(h)
        logvar = self._enc_log_sigma(h)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10, log_interval=100):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            torch.set_grad_enabled(True)
            train_loss = 0
            for batch_idx, (inputs, l_inputs) in enumerate(trainloader):
                if len(inputs.shape) > 4:
                    shp = inputs.shape
                    inputs = inputs.view(-1, shp[2], shp[3], shp[4])
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                    l_inputs = l_inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                l_inputs = Variable(l_inputs)

                outputs, mu, logvar = self.forward(inputs)
                loss = self.loss_function(outputs, l_inputs, mu, logvar)
                train_loss += loss.data * len(inputs)
                loss.backward()
                optimizer.step()
                print("    #Iter %3d: Reconstruct Loss: %.3f" % (
                    batch_idx, loss.data[0]))

            # validate
            self.eval()
            torch.set_grad_enabled(False)
            valid_loss = 0.0
            for batch_idx, (inputs, b_info) in enumerate(validloader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                outputs, mu, logvar = self.forward(inputs)

                loss = self.loss_function(outputs, inputs, mu, logvar)
                valid_loss += loss.data * len(inputs)
                # total_loss += valid_recon_loss.data[0] * inputs.size()[0]
                # total_num += inputs.size()[0]
                #print(batch_idx, loss)
                # view reconstruct
                if batch_idx % log_interval == 0:
                    n = min(inputs.size(0), 32)
                    comparison = torch.cat([inputs.view(-1, self.nChannels, self.height, self.width)[:n],
                                            outputs.view(-1, self.nChannels, self.height, self.width)[:n]])
                    save_image(comparison.data.cpu(),
                               os.path.join('results', self.name, 'reconstruct', '{}.png'.format(epoch)), nrow=n)
                    # 'results/' + self.name + '/reconstruct/reconstruction_' + str(epoch) + '.png', nrow=n)
                    #print(
                    #    strftime("%Y-%m-%d %H:%M:%S", localtime())
                    #    + '\tTrain Epoch: {} [{}/{}]\t'.format(epoch, batch_idx * len(trainloader),
                    #                                           len(self.trainloader))
                    #    + '\tTrain loss: {}, Valid Loss: {}'.format(train_loss / len(trainloader.dataset),
                    #                                                valid_loss / len(validloader.dataset)))

            # valid_loss = total_loss / total_num
            print("#Epoch %3d: Train Loss: %.5f, Valid Loss: %.5f" % (
                epoch, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

    def test(self, testloader, save_folder):
        self.eval()
        torch.set_grad_enabled(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        for batch_idx, (inputs, image_name) in enumerate(testloader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs, mu, logvar = self.forward(inputs)

            # view reconstruct
            for i in range(0, inputs.size()[0]):
                images = outputs.view(-1, self.nChannels, self.height, self.width)[i]
                if not os.path.exists(os.path.join(save_folder, image_name[i].split('/')[-2])):
                    os.makedirs(os.path.join(save_folder, image_name[i].split('/')[-2]))
                save_image(images.data.cpu(), os.path.join(save_folder, image_name[i].split('/')[-2], image_name[i].split('/')[-1]))
                # comparison = torch.cat([inputs.view(-1, self.nChannels, self.height, self.width)[i:i + n],
                #                         outputs.view(-1, self.nChannels, self.height, self.width)[i:i + n]])
                # save_image(comparison.data.cpu(),
                #            'results/' + self.name + '/reconstruct/test/rec_group_{}_{}'.format(batch_idx, i) + '.png',
                #            nrow=n)
