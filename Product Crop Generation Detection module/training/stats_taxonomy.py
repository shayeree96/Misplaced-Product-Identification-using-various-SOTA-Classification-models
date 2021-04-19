import json
import cv2
import os
import numpy as np
from tqdm import tqdm
"""
this code is reading all original annotations and get all existing type of prefixes, and print them on screen
the code is customized for walmart batch1 and batch2, may not work for other stores
fangyi april 20, 2019
"""

​
def group_by_path(annos, grouped_annos):
    global n_prod
    global n_tag
    for box in tqdm(annos):
​
        try:
            category = box['Taxonomy']
​
            ss = list(box['Attributes'].keys())[0]
​
            sub_category = box['Attributes'][ss][0]
            category = category + '_' + sub_category
            
        except:
            category = box['Taxonomy']
​
        print(category)
        if category.startswith('Prod'):
            n_prod += 1
        if category.startswith('Label'):
            n_tag += 1
        print(n_prod, n_tag)
        if category not in grouped_annos:
            
            grouped_annos[category] = 1
        else:
            grouped_annos[category] = grouped_annos[category] + 1
    # print(grouped_annos)
    return grouped_annos
​
​
if __name__ == '__main__':
    global n_prod
    global n_tag
    n_prod = 0
    n_tag = 0
    path = './' + 'json-results'
    grouped_annos = {}
​
    for i in os.listdir(path):
        print(i)
        annos = json.load(open(path+'/'+i, 'r'))
        grouped_annos = group_by_path(annos, grouped_annos)
​
    print(grouped_annos)
    for j in grouped_annos.keys():
        if j.startswith('Prod'):
            continue
        else:
            print(j, '\n')