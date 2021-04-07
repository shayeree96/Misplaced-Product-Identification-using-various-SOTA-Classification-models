# %%
import pandas as pd
import torch
import os
import pandas as pd


l=[]
target_root='/media/Cygnus/Panos_pull_azure/upc_flatten_crops_7_20200622_new_fixture_prodtype_arranged/url_folder/all/'
path2='/media/Cygnus/Panos_pull_azure/upc_flatten_crops_7_20200622_new_fixture_prodtype_arranged/crops_prime_arranged/all/' #UnCleaned
path1='/media/Cygnus/Panos_pull_azure/upc_crops_7_20200622_new_fixture_prodtype_arranged_cleaned/shayeree/Done/' #Cleaned images

for dirName1, subdirList, fileList1 in os.walk(path2):
    
    
    if 'shelves'in dirName1:
        prime_no_target=dirName1[dirName1.rfind('/')+1:dirName1.rfind('_')]
        
        target_loc=os.path.join(target_root,prime_no_target)
        if len(fileList1)>=1:
                    fileList1.sort()
        
        if os.path.isdir(target_loc):
            prime_image=os.listdir(target_loc)         

            for dirName2, subdirList, fileList2 in os.walk(path1):
                if len(fileList2)>=1:
                    fileList2.sort()
                
                if prime_no_target in dirName2: 
                    inter=[]
                    diff=[]
                    print('prime no target :',prime_no_target)
                    set1=set(fileList1)
                    set2=set(fileList2)
                    
                    inter=list(set2.intersection(set1))
                    print(len(inter))
                    
                    inter = list(map(lambda x : os.path.join(target_loc,prime_image[0])+','+ os.path.join(dirName1,x) + ',1', inter)) 
                    
                    
                    diff=list(set1.difference(set2))
                    print(len(diff))
                    diff = list(map(lambda x : os.path.join(target_loc,prime_image[0])+',' +os.path.join(dirName1,x) + ',0', diff))                 
        
                    
                    print(inter)
                    print(diff)
                    k=inter+diff
                    print(k)
                        
                   
                               
        
with open('target_hitmiss_list_finale.txt', 'w') as filehandle:
    for listitem in l:
        filehandle.write('%s\n' % listitem)  

# %%
