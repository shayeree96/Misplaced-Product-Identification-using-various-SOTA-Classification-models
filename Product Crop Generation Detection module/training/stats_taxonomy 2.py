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
global blue_price
global blue_price_OOS
global white_price
global white_price_OOS
global group_stack
global prod
global prod_ambiguous
global prod_peg
global reso

def group_by_path(annos, grouped_annos):

    blue_price=0
    blue_price_OOS=0
    white_price=0
    white_price_OOS=0
    group_stack=0
    prod=0
    prod_ambiguous=0
    prod_peg=0
    reso=' '
    
    for category,values in annos.items():
        #print('box:',type(box))
        
        reso=' '
        if category=='Label:Blue Price Label':
             blue_price+=1
        elif category=='Label:Blue Price Label--OOS':
            blue_price_OOS+=1
        elif category=='Label:White Price Label':
            white_price+=1
        elif category=='Label:White Price Label--OOS':  
             white_price_OOS+=1
        elif category=='Product Group:Stack':      
             group_stack+=1
        elif category=='Product:Product':      
             prod+=1 
        elif category=='Product:Product--Ambiguous':      
             prod_ambiguous+=1      
        elif category=='Product:Product--Peg':      
             prod_peg+=1 
        elif category=='Department':
            continue
        else:
            reso+=str(values)
            
        '''
        print('box:',box)

        try:
            category = box['Taxonomy']
            print('in try',category)

            ss = list(box['Attributes'].keys())[0]

            #sub_category = box['Attributes'][ss][0]
            #category = category + '_' + sub_category
            
        except:
            category = box['Product:Product--Peg']

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
    
    
    return category
    

'''
if __name__ == '__main__':
    global n_prod
    global n_tag
    n_prod = 0
    n_tag = 0
    path = './' + 'json-results'
    grouped_annos = {}

    for i in os.listdir(path):
        annos = json.load(open(path+'/'+i, 'r'))
        grouped_annos = group_by_path(annos, grouped_annos)
    for key,value in annos.items():
        if type(value)==list:
            print('Category and length :',key,len(value))
    
    #print(grouped_annos)
    