import json
import cv2
import os
import numpy as np
from tqdm import tqdm



def group_by_path(annos):


    grouped_annos = {}
    for box in tqdm(annos):
        folder = box['File Folder']
        filename = box['Filename']
        if folder.startswith('BNR'):
            1
        else:
            folder = filename[:16] + '/' + folder
        if filename[-4:] != '.jpg':
            return None, 1
        path = os.path.join(folder, filename)

        if path not in grouped_annos:
            grouped_annos[path] = {'Resolution': box['Resolution']}
        try:
            category = box['Taxonomy']
            try:
                sub_category = box['Attributes']['Label Type'][0]
                #print(sub_category)
                category = category + '_' + sub_category
            except:
                sub_category = box['Attributes']['Product Type'][0]
                category = category + '_' + sub_category
        except:
            category = box['Taxonomy']
        x, y, w, h = box['Left'], box['Top'], box['Width'], box['Height']
        if category not in grouped_annos[path]:
            grouped_annos[path][category] = [[x, y, x + w - 1, y + h - 1]]
        else:
            grouped_annos[path][category].append([x, y, x + w - 1, y + h - 1])
    return grouped_annos


def filter_instance(grouped_annos, list_ignore, list_mask):
    in_dir = './data'
    out_dir = './processed'
    json_dir = './json'

    for path in tqdm(grouped_annos):
        im = cv2.imread(os.path.join(in_dir, path))
        if im is None:
            print(os.path.join(in_dir, path))
            continue
        anno = grouped_annos[path]
        # remove annos of some categories
        for cat in list_ignore:
            if cat in anno:
                del anno[cat]
        # remove annos of some categories and mask out corresponding regions
        mask_boxes = []
        for cat in list_mask:
            if cat in anno:
                mask_boxes.extend(anno[cat])
                del anno[cat]
        im = mask_image(im, mask_boxes, color=(104, 116, 124))

        # output both image and json annotation
        image_path = os.path.join(out_dir, path)
        im_folder = os.path.dirname(image_path)
        if not os.path.exists(im_folder):
            os.makedirs(im_folder)
        cv2.imwrite(image_path, im)
        json_path = os.path.join(json_dir, path.replace('.jpg', '.json'))
        json_folder = os.path.dirname(json_path)
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
        with open(json_path, "w") as f:
            json.dump(anno, f, indent=4)


def mask_image(im, mask_boxes, color):
    for [x1, y1, x2, y2] in mask_boxes:
        im[y1:y2 + 1, x1:x2 + 1, :] = np.array(color).reshape((1, 1, 3)).astype(im.dtype)
    return im


if __name__ == '__main__':

    path = './' + 'json-results'
    for i in os.listdir(path):

        annos = json.load(open(path+'/'+i, 'r'))
        grouped_annos = group_by_path(annos)
        if 0:
            print(i)
            continue
        else:
            with open('./grouped_annos_new.json', 'w') as f:
                json.dump(grouped_annos, f, indent=4)

            list_ignore = ['Section:Shelf Marking', 'Product Group:Retail Box','Products:Retail Box', 'Products:Empty Box',]
            list_mask = ['Products:Stack with Grill', 'Section:Ambiguous', 'Products:Ambiguous',
                         'Product:Product_Grill', 'Products:Product with Grill',
                         'Product:Ambiguous',
                         ]
            filter_instance(grouped_annos, list_ignore, list_mask)



