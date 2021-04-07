# %%
!nvidia-smi

# %%
######### IMPORTING NECESSARY MODULES #########
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm


# %%
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
"""
**Dataloading Scheme**
"""

# %%
trainlist = 'training_list_4_departments.txt'
validlist ='validation_list_4_departments.txt'


# %%
# Create a mapping of Unique product Ids to Labels(0 to 31127 classes)
# output = dictionary containing mapping of each upc to a label from (0 to 31127)  

with open(trainlist, mode = 'r') as f:
    
    Y=[]
    for line in f:
        path, UPC = line[:-1].split(',')

        Y.append(UPC)
        
prime_number_list = sorted(set(Y))

prime_number_dict = { prime_number_list[i] :i for i in range(0, len(prime_number_list) ) }

# %%
len(prime_number_dict)

# %%
class mydataset():    

    def __init__(self, classification_list, prime_number_dict, name):

        super(mydataset).__init__()
        
        self.X = []
        self.Y = []
        
        with open(classification_list, mode = 'r') as f:
            
            for line in f:
                path, Prime_Number = line[:-1].split(',')

                self.X.append(path)
                self.Y.append(prime_number_dict[Prime_Number])
        

        if name == 'valid':
            self.transform = transforms.Compose([   
#                                                     transforms.RandomResizedCrop(224),
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([   transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                                                            ])
    
    def __getitem__(self,index):
        
        image = self.X[index]        
        label = float(self.Y[index])
        
        image = (Image.open(image))
               
        image = self.transform(image)
        
        return image, torch.as_tensor(label).long()
        
  
    def __len__(self):
        return len(self.X)

# %%
# #### Train Dataloader #### 
train_dataset = mydataset(trainlist, prime_number_dict, name='train')          
train_dataloader = data.DataLoader(train_dataset, shuffle= True, batch_size = 128, num_workers=16,pin_memory=True)


#### Validation Dataloader #### 
validation_dataset = mydataset(validlist, prime_number_dict, name='valid')         
validation_dataloader = data.DataLoader(validation_dataset, shuffle=False, batch_size = 128, num_workers=16,pin_memory=True)

# %%
"""
**RESNET Architecture**
"""

# %%
"""
**Model Definition**
"""

# %%
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes, bottleneck=False):
        super(ResNet, self).__init__()        
        
        
        blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
        layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
        assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(blocks[depth], 64, layers[depth][0])
        self.layer2 = self._make_layer(blocks[depth], 128, layers[depth][1], stride=2)
        self.layer3 = self._make_layer(blocks[depth], 256, layers[depth][2], stride=2)
        self.layer4 = self._make_layer(blocks[depth], 512, layers[depth][3], stride=2)
        self.avgpool = nn.AvgPool2d(7) 
        self.fc = nn.Linear(512 * blocks[depth].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
    
        return x


# %%
"""
**Initialize the model**
"""

# %%
model = ResNet(depth = 50, num_classes = 12828)
model = nn.DataParallel(model,device_ids=[6,7]).to(device)
model

# %%


# %%


# %%


# %%
"""
**Helper function for Cutmix
https://arxiv.org/pdf/1905.04899v2.pdf**
"""

# %%
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# %%


# %%


# %%
"""
**Training Method**
"""

# %%
def train(model, data_loader, test_loader,beta, cutmix_prob, epochs):
    model.train()

    for epoch in range(epochs):
        avg_loss = 0.0
                
        
        for batch_num, (feats, target) in enumerate(data_loader):
            feats, target = feats.to(device), target.to(device)
            
            
            r = np.random.rand(1)
            if beta > 0 and r < cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(feats.size()[0]).to(device)
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(feats.size(), lam)
                feats[:, :, bbx1:bbx2, bby1:bby2] = feats[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (feats.size()[-1] * feats.size()[-2]))
                # compute output
                output = model(feats)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
            else:
                # compute output
                output = model(feats)
                loss = criterion(output, target)


                                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

#             if batch_num % 100 == 99:
#                 print('loss', avg_loss/100)

            del feats
            del target
            del loss

        lr_scheduler.step()

        print('Epoch: ', epoch+1)

        print('training loss = ', avg_loss/len(data_loader))
        train_loss.append(avg_loss/len(data_loader))

        ## Check performance on validation set after an Epoch
        valid_loss, valid_acc = test_classify(model, test_loader)
        print('Val Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(valid_loss, valid_acc))
        v_loss.append(valid_loss)
        v_acc.append(valid_acc)

    
        
        
        #########save model checkpoint #########
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Training_Loss_List':train_loss,
            'Validation_Loss_List':v_loss,
            'Validation_Accuracy_List': v_acc,
            'Epoch':epoch
            'lr_scheduler': lr_scheduler.state_dict() 

            }, 'saved_model_checkpoints/cutmix_2gpu')


def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total



# %%
"""
**Hyperparameters**
"""

# %%
# # Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 30, gamma = 0.1)


# Epochs
num_Epochs = 120

beta=1

cutmix_prob = 1

# %%
train_loss= []
v_loss = []
v_acc = []

# %%


# %%
"""
**Train the model**
"""

# %%
train(model, train_dataloader, validation_dataloader, beta, cutmix_prob, epochs = num_Epochs)

# %%


# %%
"""
**Load saved model from checkpoint**
"""

# %%
checkpoint = torch.load('saved_model_checkpoints/cutmix_2gpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
train_loss = checkpoint['Training_Loss_List'] 
v_loss = checkpoint['Validation_Loss_List']
v_acc = checkpoint['Validation_Accuracy_List']
epoch = checkpoint['epoch']
lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


# %%


# %%


# %%
"""
**Generate plots**
"""

# %%
# plt.figure(figsize=(8,8))
# x = np.arange(91,121)
# plt.plot(x, train_loss)
# plt.xlabel('Epochs', fontsize =16)
# plt.ylabel('Training Loss', fontsize =16)
# plt.title('Training Loss v/s Epochs',fontsize =16)


plt.figure(figsize=(8,8))
x = np.arange(1,128)
plt.plot(x, train_loss[:-1], label = 'Training Loss')
plt.plot(x, v_loss, label = 'Validation Loss')
plt.xlabel('Epochs', fontsize =16)
plt.ylabel('Loss', fontsize =16)
plt.title('Loss v/s Epochs',fontsize =16)
plt.legend(fontsize=16)

# %%


# %%
