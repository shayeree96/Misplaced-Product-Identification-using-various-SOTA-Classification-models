from __future__ import print_function

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

#from apex import scaler
import os
import argparse
import numpy as np
import math

import torch
#from torch.cuda import amp
import torch.optim as optim
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet, TagNet
from datagen import NRF

from torch.autograd import Variable
print(APEX_AVAILABLE)
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='resnet50', type=str, help='backbone network')
parser.add_argument('--ckpt', default='Retina50ProdB1', type=str, help='checkpoint folder name')
parser.add_argument('--type', default='tag', type=str, help='Choose "tag" or "prod"')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = NRF(root='./data', image_set=args.type, transform=transform, input_size=768)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4, collate_fn=trainset.collate_fn)

ckpt_dir = os.path.join('./checkpoint_list_wise/albertsons', args.ckpt)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Model
print('==> Setting up network..')
if args.type == 'prod':
    print('im gonno use Retina_Net')
    net = RetinaNet(num_classes=1, num_anchors=15, backbone=args.net)
elif args.type == 'tag':
    net = TagNet(num_classes=1, num_anchors=9)
else:
    raise TypeError('Unknown detection type, choose "prod" or "tag"')

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('/media/Darius/shayeree/training/checkpoint_list_wise/albertsons/Retina50ProdB1/ckpt_0005_0.1237.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    params_dict = torch.load('./model/{:s}.pth'.format(args.net))
    net_dict = net.fpn.state_dict()
    params_dict = {k: v for k, v in params_dict.items() if k in net_dict}
    net_dict.update(params_dict)
    net.fpn.load_state_dict(net_dict)
#net,optimizer=amp.initialize(net,optimizer,opt_level='01',loss_scale="dynamic")
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss(num_classes=1)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
amp_handle = amp.init(enabled=True, verbose=True)
optimizer = amp_handle.wrap_optimizer(optimizer)


#net,optimizer=amp.initialize(net.cuda(),optimizer,opt_level='O1',loss_scale="dynamic")
#net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#net=apex.parallel.DistributedDataParallel(net, device_ids=range(torch.cuda.device_count()))

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)
# Training

def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
   # net.module.freeze_bn()
    train_loss = 0

    if epoch==0:
        lr=1e-5

    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        print('E: %d, B: %d / %d' % (epoch, batch_idx+1, len(trainloader)), end=' | ')
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        with torch.no_grad():
         with optimizer.scale_loss(loss) as scaled_loss:
                scaled_loss.backward()
         optimizer.step()
       # loss.backward()

        torch.nn.utils.clip_grad_norm(net.parameters(),0.25)

        train_loss += scaled_loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f' % (scaled_loss.data[0], train_loss/(batch_idx+1)))

    if (epoch + 1) % 5 == 0:
        print('Saving..')
        loss = train_loss / len(trainloader)
        state = {
            'net': net.module.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(ckpt_dir, 'ckpt_{:04d}_{:.4f}.pth'.format(epoch+1, loss)))


for epoch in range(start_epoch, start_epoch + 200):
    if epoch<=10:
         train(epoch)



