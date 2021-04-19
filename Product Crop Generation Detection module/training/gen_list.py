import json
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil



path = './data'
for i in os.listdir(os.path.join(path, 'processed')):
    for j in os.listdir(os.path.join(os.path.join(path, 'processed'), i)):
        cur_path = os.path.join(i, j)

        if cur_path[5:7] == 'Pa':
            for k in os.listdir(path + '/' + 'processed' + '/' + cur_path):
                ppp = os.path.join(cur_path, k)
                #print(ppp[:-4])

                if not os.path.exists((os.path.join(path, 'json') + '/' + ppp[:-4] + '.json')):
                    print((os.path.join(path, 'json') + '/' + ppp[:-4] + '.json'))
                    print(1)
        else:
            1
            print(cur_path[:-4])




            if not os.path.exists(os.path.join(os.path.join(path, 'json'), i) + '/' + j[:-4] + '.json'):
                print(os.path.join(os.path.join(path, 'json'), i) + '/' + j[:-4] + '.json')
                print(2)






