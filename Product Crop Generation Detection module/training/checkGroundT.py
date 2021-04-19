import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from PIL import Image
import json
import os
from PIL import Image
import random
import datetime
import os
import numpy as np
import cv2
import tqdm

root = './data'
print('Im in root',root)
list_file = os.path.join(root, 'trainlist.txt')
with open(list_file) as f:
    lines = f.readlines()

for line in lines:
    index = line.strip()
    anno = json.load(open(os.path.join(root, 'json', index + '.json'), 'r'))
    img = Image.open(os.path.join(root, 'processed', index + '.jpg'))

    fig, ax = plt.subplots(1)


    # root = 'BNR-bossanova_riverside-20181218_183345UTC-atlasoscar11-NRF_Demo-Panorama'
    #img_path = './newNRF_60images/processed/' + root + '/' + root + '.jpg'

    #img = Image.open(img_path)
    ax.imshow(img)
    #anno = json.load(open('./newNRF_60images/json/' + root + '/' + root + '.json', 'r'))
    extend = 0
    box = []
    box1 = []

    for cat in anno:
        if cat.startswith('Product') :
            print(cat)
            box.extend(anno[cat])

        for bb in box:
            x1 = bb[0]
            y1 = bb[1]
            w = bb[2] - bb[0] + 1
            h = bb[3] - bb[1] + 1

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

    for cat in anno:
        if cat.startswith('Label:'):
            print(cat)
            box1.extend(anno[cat])

        for bb in box1:
            x1 = bb[0]
            y1 = bb[1]
            w = bb[2] - bb[0] + 1
            h = bb[3] - bb[1] + 1

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()

    # cv2.imwrite('/home/fangyi'+'/AF01_color_panorama.jpg', img)
    '''mask'''
    '''
    img = cv2.imread(img_path)
    for cat in anno:
        if cat.startswith('Section:Ambiguous'):
            print(cat)
            box.extend(anno[cat])

        for bb in box:
            x1 = bb[0]
            y1 = bb[1]
            w = bb[2] - bb[0] + 1
            h = bb[3] - bb[1] + 1

            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

            color=(104, 116, 124)

            img[y1:y1+h, x1:x1+w, :] = np.array(color).reshape((1, 1, 3)).astype(img.dtype)
    
    cv2.imwrite('./datanew/processed/'+root+'/AF01_color_panorama.jpg', img)
    '''



