'''Perform transforms on both PIL image and object boxes.'''
import math
import random

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw


def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BILINEAR), \
           boxes*torch.Tensor([sw,sh,sw,sh])

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x,y,x,y])
    boxes[:,0::2].clamp_(min=0, max=w-1)
    boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes

def random_sample_crop(img, boxes, labels):
    width, height = img.size
    short_edge = min(width, height)
    sample_options = [0, 1, 1, 1, 1]
    num_trails = 25
    for _ in range(num_trails):
        opt = random.choice(sample_options)
        if opt == 0:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        w = h = int(round(short_edge * scale))
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        rect = (x, y, x + w, y + h)
        keep = check_inside(boxes, rect)
        if len(keep) > 0:
            boxes = boxes[keep, :]
            boxes -= torch.Tensor([x, y, x, y])
            boxes[:, 0::2].clamp_(min=0, max=w - 1)
            boxes[:, 1::2].clamp_(min=0, max=h - 1)
            img_crop = img.crop(rect)
            labels = labels[keep]
            return img_crop, boxes, labels
    # if not success, return original
    return img, boxes, labels

def random_fixed_crop(img, boxes, labels, size):
    width, height = img.size
    w, h = size
    while 1:
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)
        rect = (x, y, x + w, y + h)
        keep = check_inside(boxes, rect)
        if len(keep) > 0:
            boxes = boxes[keep, :]
            boxes -= torch.Tensor([x, y, x, y])
            boxes[:, 0::2].clamp_(min=0, max=w - 1)
            boxes[:, 1::2].clamp_(min=0, max=h - 1)
            img_crop = img.crop(rect)
            labels = labels[keep]
            return img_crop, boxes, labels

def check_inside(boxes, crop):
    x_ctr = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2.0
    valid = (x_ctr > crop[0]) & (x_ctr < crop[2]) & (y_ctr > crop[1]) & (y_ctr < crop[3])
    return valid.nonzero().view(-1)

def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).

    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    boxes -= torch.Tensor([j,i,j,i])
    boxes[:,0::2].clamp_(min=0, max=ow-1)
    boxes[:,1::2].clamp_(min=0, max=oh-1)
    return img, boxes

def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.

    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return img, boxes

def draw(img, boxes):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.show()


def test():
    img = Image.open('./image/000001.jpg')
    boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    img, boxes = random_crop(img, boxes)
    print(img.size)
    draw(img, boxes)

# test()
if __name__ == '__main__':
    # img = Image.open('./image/000001.jpg')
    # boxes = torch.Tensor([[48, 240, 195, 371], [8, 12, 352, 498]])
    # img, boxes = random_sample_crop(img, boxes)
    # print(img.size)
    # draw(img, boxes)
    img = Image.open('./image/pano.jpg')
    print(img.size)
    width, height = img.size
    w = h = 512
    x = 0
    y = 3000
    crop = img.crop((x, y, x + w, y + h))
    print(crop.size)
    print(img.size)
    # for _ in range(100):
    #     x = random.randint(0, width - w)
    #     y = random.randint(0, height - h)
    #     rect = (x, y, x + w, y + h)
    #     img = img.crop(rect)
    #     print img.size
