'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

import os
import sys
import json
import random
import xml.etree.ElementTree as ET

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop, random_sample_crop, random_fixed_crop


class WalmartTag(data.Dataset):
    def __init__(self, root, image_set, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          image_set: (str) train, val or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.input_size = input_size

        self.images = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder(areas=[50*50., ], aspect_ratios=[2/1., ], scale_ratios=[1.0, ], init_feat_stride=16.)

        list_file = os.path.join(root, 'tag_{:s}.txt'.format(image_set))
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            index = line.strip()
            anno_file = os.path.join(root, 'data', 'anno_tags', index + '.xml')
            tree = ET.parse(anno_file)
            objs = tree.findall('object')
            num_boxes = len(objs)
            if num_boxes == 0:
                continue
            box = []
            label = []
            for obj in objs:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) - 1
                ymin = float(bbox.find('ymin').text) - 1
                xmax = float(bbox.find('xmax').text) - 1
                ymax = float(bbox.find('ymax').text) - 1
                c = 0
                box.append([xmin, ymin, xmax, ymax])
                label.append(int(c))
            img = Image.open(os.path.join(root, 'data', 'panos', index + '.jpg'))
            crop = img.crop((0, 0, 100, 100))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            self.images.append(img)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_images = len(self.images)
        self.num_samples = self.num_images * 256

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        idx = random.choice(range(self.num_images))
        img = self.images[idx]
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.input_size

        # Data augmentation.
        if self.image_set == 'train':
            # img = photometric_distort(img)
            img_crop, boxes, labels = random_fixed_crop(img, boxes, labels, (size, size))

        img_crop = self.transform(img_crop)
        return img_crop, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


class WalmartProduct(data.Dataset):
    def __init__(self, root, image_set, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          image_set: (str) train, val or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.input_size = input_size

        self.images = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        list_file = os.path.join(root, 'product_{:s}.txt'.format(image_set))
        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            index = line.strip()
            anno_file = os.path.join(root, 'data', 'anno_products', index + '.xml')
            tree = ET.parse(anno_file)
            objs = tree.findall('object')
            num_boxes = len(objs)
            if num_boxes == 0:
                continue
            box = []
            label = []
            for obj in objs:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) - 1
                ymin = float(bbox.find('ymin').text) - 1
                xmax = float(bbox.find('xmax').text) - 1
                ymax = float(bbox.find('ymax').text) - 1
                c = 0
                box.append([xmin, ymin, xmax, ymax])
                label.append(int(c))
            img = Image.open(os.path.join(root, 'data', 'panos', index + '.jpg'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            crop = img.crop((0, 0, 100, 100)) # preload image
            self.images.append(img)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_images = len(self.images)
        self.num_samples = self.num_images * 256

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        idx = random.choice(range(self.num_images))
        img = self.images[idx]
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.input_size

        # Data augmentation.
        if self.image_set == 'train':
            # img = photometric_distort(img)
            img_crop, boxes, labels = random_fixed_crop(img, boxes, labels, (size, size))

        img_crop = self.transform(img_crop)
        return img_crop, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


class WalmartBatch1(data.Dataset):
    def __init__(self, root, image_set, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          image_set: (str) train, val or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.input_size = input_size

        self.images = []
        self.boxes = []
        self.labels = []

        if self.image_set == 'prod':
            self.encoder = \
                DataEncoder(areas=[32*32., 64*64., 128*128., 256*256., 512*512.],
                            aspect_ratios=[1/4., 1/2., 1/1., 2/1., 4/1.],
                            scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
                            init_feat_stride=8.)

            self.prefix = 'Products:'
        elif self.image_set == 'tag':
            self.encoder = \
                DataEncoder(areas=[40.69**2, ], aspect_ratios=[1.8, ], scale_ratios=[1.0, ], init_feat_stride=16.)
            self.prefix = 'Labels:'
        else:
            raise TypeError('Unknown detection type, choose "prod" or "tag"')

        list_file = os.path.join(root, 'trainlist.txt')
        with open(list_file) as f:
            lines = f.readlines()

        for line in tqdm(lines):
            index = line.strip()
            anno = json.load(open(os.path.join(root, 'json', index + '.json'), 'r'))
            box = []
            label = []
            for cat in anno:
                if cat.startswith(self.prefix):
                    box.extend(anno[cat])
                    c = 0
                    label.extend([int(c)] * len(anno[cat]))
            img = Image.open(os.path.join(root, 'processed', index + '.jpg'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            crop = img.crop((0, 0, 100, 100))  # preload image
            self.images.append(img)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_images = len(self.images)
        self.num_samples = self.num_images * 256

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        idx = random.choice(range(self.num_images))
        img = self.images[idx]
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.input_size

        # Data augmentation.
        img_crop, boxes, labels = random_fixed_crop(img, boxes, labels, (size, size))

        img_crop = self.transform(img_crop)
        return img_crop, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


class WiderFace(data.Dataset):
    def __init__(self, root, image_set, transform, input_size, encoder):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          image_set: (str) train, val or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = encoder

        list_file = os.path.join(root, 'wider_face_split', 'wider_face_{:s}_list.txt'.format(image_set))
        with open(list_file) as f:
            lines = f.readlines()
            # self.num_samples = len(lines)

        for line in lines:
            index = line.strip()
            anno_file = os.path.join(self.root, 'Annotations', index + '.xml')
            tree = ET.parse(anno_file)
            objs = tree.findall('object')
            valid_objs = [obj for obj in objs if int(obj.find('invalid').text) == 0]
            objs = valid_objs
            num_boxes = len(objs)
            if num_boxes == 0:
                continue
            box = []
            label = []
            for obj in objs:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) - 1
                ymin = float(bbox.find('ymin').text) - 1
                xmax = float(bbox.find('xmax').text) - 1
                ymax = float(bbox.find('ymax').text) - 1
                c = 0
                box.append([xmin, ymin, xmax, ymax])
                label.append(int(c))
            self.fnames.append(index + '.jpg')
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, 'WIDER_' + self.image_set, 'images', fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.input_size

        # Data augmentation.
        if self.image_set == 'train':
            # img = photometric_distort(img)
            img, boxes, labels = random_sample_crop(img, boxes, labels)
            img, boxes = random_flip(img, boxes)
            img, boxes = resize(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='/mnt/hgfs/D/download/PASCAL_VOC/voc_all_images',
                          list_file='./data/voc12_train.txt', train=True, transform=transform, input_size=600)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

# test()


class NRF(data.Dataset):
    def __init__(self, root, image_set, transform, input_size):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          image_set: (str) train, val or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
        '''
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.input_size = input_size

        self.images = []
        self.boxes = []
        self.labels = []

        if self.image_set == 'prod':
            self.encoder = \
                DataEncoder(areas=[32*32., 64*64., 128*128., 256*256., 512*512.],
                            aspect_ratios=[1/4., 1/2., 1/1., 2/1., 4/1.],
                            scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
                            init_feat_stride=8.)

            self.prefix = 'Product'
        elif self.image_set == 'tag':
            self.encoder = \
                DataEncoder(areas=[40.0**2, ], aspect_ratios=[1.3, 2.4, 3.45,], scale_ratios=[1.0, 1.3, 1.6], init_feat_stride=16.)
            self.prefix = 'Label'
        else:
            raise TypeError('Unknown detection type, choose "prod" or "tag"')

        list_file = os.path.join(root, 'trainlist')
        with open(list_file) as f:
            lines = f.readlines()

        for line in tqdm(lines):
            index = line.strip()
            anno = json.load(open(os.path.join(root, 'json', index + '.json'), 'r'))
            box = []
            label = []
            for cat in anno:
                if cat.startswith(self.prefix):
                    box.extend(anno[cat])
                    c = 0
                    label.extend([int(c)] * len(anno[cat]))
            img = Image.open(os.path.join(root, 'processed', index + '.jpg'))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            crop = img.crop((0, 0, 100, 100))  # preload image
            self.images.append(img)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_images = len(self.images)
        self.num_samples = self.num_images * 256

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        idx = random.choice(range(self.num_images))
        img = self.images[idx]
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        size = self.input_size

        # Data augmentation.
        img_crop, boxes, labels = random_fixed_crop(img, boxes, labels, (size, size))

        img_crop = self.transform(img_crop)
        return img_crop, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples
