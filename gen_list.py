import json
import os
import numpy as np
from tqdm import tqdm
import shutil
from random import shuffle
t = []

path = '/media/Darius/Panos_pull_azure/upc_flatten_crops_7_20200622_new_fixture_prodtype_arranged_AIcleaned_crops/crops_prime_arranged/all'
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path, i)):
        t.append(os.path.join(path, i, j) + '\n')

shuffle(t)
a = t[0:1000]
t = t[1000:100000]
with open('./trainlist.txt', 'w') as f:

    f.writelines(["%s" % item  for item in t])

with open('./testlist.txt', 'w') as ff:

    ff.writelines(["%s" % item  for item in a])
