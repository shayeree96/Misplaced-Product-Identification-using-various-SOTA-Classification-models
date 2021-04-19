import argparse
import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from retinanet import RetinaNet, TagNet, ProductNet
from encoder import DataEncoder
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import json
import numpy as np
from utils import *
import csv
from tqdm import tqdm
import pprint
import shutil


def _SearchObjectSingle(annos, all_categories, target, tgt_loc):
    '''For a target class, search store the objects close to it.
        return {'cat_1': [[]],
                'cat_2': [[]]}
    '''
    pp = pprint.PrettyPrinter()
    tgt_loc = np.array(tgt_loc)
    l, t, r, b = tgt_loc
    vicinityPoints = {}
    for category in annos:
        if category in all_categories and category != target:
            # Search products within retail boxes
            boxes = np.array(annos[category])
            boxesCtr = np.stack([0.5*(boxes[:,0]+boxes[:,2]), 0.5*(boxes[:,1] + boxes[:,3])], axis=1)
            mask_x = np.logical_and(boxesCtr[:,0]>= l, boxesCtr[:,0]<=r)
            mask_y = np.logical_and(boxesCtr[:,1]>= t, boxesCtr[:,1]<=b)
            mask = np.logical_and(mask_x, mask_y)
            vicinityPoints[category] = mask
    return vicinityPoints


def mergeObject(annos, matchedPoints, target):
    merged = {}
    for group in matchedPoints:
        for cat in group:
            if cat != target:
                if cat not in merged:
                   merged[cat] = [group[cat]]
                else:
                    merged[cat].append(group[cat])
    for cat in merged:
        merged[cat] = np.logical_or.reduce(merged[cat])
    return merged


def _SearchObject(annos, all_categories, target):
    matchedPoints = []
    if target not in annos or annos.get(target) is None:
        return {}
    for tgt_loc in annos[target]:
        matchedPoints.append(_SearchObjectSingle(annos, all_categories, target, tgt_loc))
    matchedPoints = mergeObject(annos, matchedPoints, target)
    return matchedPoints


parser = argparse.ArgumentParser(description='PyTorch Walmart Testing')
parser.add_argument('--thresh', default=0.40, type=float, help='Confidence threshold')
parser.add_argument('--scale', default=1.0, type=float, help='Testing scale')
parser.add_argument('--ckpt_dir', default='/media/Darius/shayeree/mixed_precision/training/checkpoint_list_wise/albertsons/Retina50ProdB1', type=str, help='checkpoint directory')
parser.add_argument('--out', default='/media/Darius/shayeree/mixed_precision/training/results_quantative/list2.csv', type=str)
parser.add_argument('--chunk_overlap', default=256, type=int, help='overlap x of each chunk')
args = parser.parse_args()

# all_categories = json.load(open(args.all_categories, 'r'))
if not os.path.exists(os.path.dirname(args.out)):
    os.makedirs(os.path.dirname(args.out))

header = ['Ckpt',
          'Chunk Size',
          'Threshold',
          'Chunk Overlap',
          'Num Imgs',
          'Avg Product Precision (%)',
          'Avg Product Recall (%)',
          'Avg fp/image']

with open(args.out, 'a', newline='') as fsd:
    fsd_writer = csv.DictWriter(fsd, fieldnames=header, delimiter='\t')
    fsd_writer.writeheader()

pp = pprint.PrettyPrinter()
# pp.pprint(all_categories)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

#v23
# prod_encoder = DataEncoder(areas=[32*32., 64*64., 128*128., 256*256., 512*512.],
#                       aspect_ratios=[1/8, 1/4., 1/2., 1/1., 2/1., 4/1., 8/1],
#                       scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
#                       init_feat_stride=8.)
#v20
prod_encoder = DataEncoder(areas=[32*32., 64*64., 128*128., 256*256., 512*512.],
                      aspect_ratios=[1/4., 1/2., 1/1., 2/1., 4/1.],
                      scale_ratios=[1., pow(2, 1/3.), pow(2, 2/3.)],
                      init_feat_stride=8.)

# ckpts = os.listdir(args.ckpt_dir)
# ckpts = ['ckpt_0060_0.0661.pth']
ckpts = ['ckpt_0017_10540.0801.pth']

# ckpts = ['ckpt_0022_0.0131.pth', 'ckpt_0026_0.0127.pth', 'ckpt_0030_0.0125.pth']

#v20
net = RetinaNet(num_classes=1, num_anchors=15)
# #v23
# net = RetinaNet(num_classes=1, num_anchors=21)
# net = TagNet(num_classes=1, num_anchors=9)

for ckpt in ckpts:

    net.load_state_dict(torch.load(os.path.join(args.ckpt_dir, ckpt))['net'])
    net.eval()
    net.cuda()

    chunk_sizes = [4096]
    threshes =  [0.5]
    overlaps = [256]
    experiment = 1
    for chunk_size in chunk_sizes:
        for thresh in threshes:
            for overlap in overlaps:
                experiment += 1
                print("="*200)

                # path define
                data_dir = './raw/'
                before_processing = './raw/no_department/'
                num_images = 0
                num_products = 0
                num_det = 0
                num_tp_products = 0
                num_rc_products = 0
                num_false = 0
                avg_precision = 0.0
                avg_recall = 0.0

                with open(data_dir+'rawlist.txt','r') as f:
# ====================================================== Per File Level ======================================================== #
                    pbar = tqdm(f, desc="ckpt: {}, conf_thresh: {}".format(ckpt, thresh), dynamic_ncols=True)
                    for line in pbar:

                        num_objects = 0
                        line = line.strip('\r\n')
                        json_file = data_dir + 'json/' + line + '.json'
                        im_file = before_processing + line + '.jpg'
                        print(json_file, im_file)
                        if not os.path.exists(json_file) or not os.path.exists(im_file):
                            # print("Json file not exist")
                            continue

                        # prepare groundtruth
                        with open(json_file, 'r') as each_json:
                            datastore = json.load(each_json)

                            gt_boxes = np.array([y for x in datastore.keys() if 'Product' in x or 'Products' in x for y in np.array(datastore[x])])
                            num_objects += gt_boxes.shape[0]
                            if len(gt_boxes) == 0:
                                print("No valid annotation found, skip to the next image.")
                                continue

                            labels = np.ones(num_objects, dtype=int)
                        gt_boxes = Variable(torch.from_numpy(gt_boxes).float())
                        labels = Variable(torch.from_numpy(labels).float())

# ===================================================================== prepare image ===================================================================== #
                        img = Image.open(im_file)
                        w, h = img.size

                        w_new = int(round(w * args.scale))
                        h_new = int(round(h * args.scale))
                        img = img.resize((w_new, h_new), resample=Image.BILINEAR)

                        chunk_stride = chunk_size - overlap
                        chunk_starts = list(range(0, w_new - chunk_size, chunk_stride))
                        if chunk_starts == []:
                            chunk_starts = [0]

                        if chunk_starts[-1] != w_new - chunk_size:
                            chunk_starts.append(w_new - chunk_size)

                        # margin = 50
                        all_boxes = torch.zeros(0, 4)
                        all_scores = torch.zeros(0)

                        for x1 in chunk_starts:
                            x2 = x1 + chunk_size
                            s1 = x1
                            s2 = x2
                            img_crop = img.crop((x1, 0, x2, h_new))
                            x = transform(img_crop)
                            x = x.unsqueeze(0)
                            x = x.cuda()
                            with torch.no_grad():
                                loc_preds, cls_preds = net(x)

                            boxes, scores = prod_encoder.decode_face(loc_preds.cpu().data.squeeze(),
                                                                cls_preds.cpu().data.squeeze(), (x2 - x1, h_new), conf_thresh=thresh)
                            if boxes is None:
                                # print(line)
                                continue
                            boxes[:, 0::2] += x1
                            keep = torch.squeeze(torch.nonzero((boxes[:, 2] > s1) & (boxes[:, 0] < s2)))
                            boxes = boxes[keep]
                            scores = scores[keep]
                            boxes = boxes / args.scale

                            if len(boxes.shape) == 1:
                                boxes = boxes.view(1,4)
                                scores = scores.view(-1)

                            all_boxes = torch.cat((all_boxes, boxes), dim=0)
                            all_scores = torch.cat((all_scores, scores))

                        # perform nms first on all predictions
                        keep = box_nms(all_boxes, all_scores)
                        pred_boxes = all_boxes[keep]
                        scores = all_scores[keep]

                        # # all evalulations
                        if len(pred_boxes) == 0:
                            print('No predictions been made, skip to the next image')
                            print(line)
                            continue
                        num_images += 1
# ================================================================== remove ground truth overlaps ================================================================= #
                        keep_gt = box_nms(gt_boxes, labels)
                        gt_boxes = gt_boxes[keep_gt]
                        labels = labels[keep_gt]
                        # if len(within_boxes) and len(within_labels):
                        #     keep_retail_gt = box_nms(within_boxes, within_labels)
                        #     within_boxes = within_boxes[keep_retail_gt]
                        #     within_labels = within_labels[keep_retail_gt]

# ======================================================= search all predictions within ground truth retail boxes ================================================= #
                        # retail_boxes = pred_boxes.cpu().numpy()
                        # retail_scores = scores.cpu().numpy()
                        # temp_datastore = {}
                        # temp_datastore[args.target] = datastore.get(args.target, None)
                        # temp_datastore.update({'Product:Product':retail_boxes})
                        # pred_mask = _SearchObject(temp_datastore, all_categories, args.target)
                        # retail_boxes = torch.from_numpy(retail_boxes[pred_mask.get('Product:Product', np.zeros(len(retail_boxes), dtype=bool))])
                        # retail_scores = torch.from_numpy(retail_scores[pred_mask.get('Product:Product', np.zeros(len(retail_boxes), dtype=bool))])

# ===================================================== detect precision and recall of products in a single image ================================================= #

                        iou = box_iou(gt_boxes, pred_boxes)
                        true_detect_precision = sum([(iou[:,k]>=0.5).any().item() for k in range(iou.shape[1])])
                        true_detect_recall = sum([(iou[k]>=0.5).any().item() for k in range(iou.shape[0])])
                        false_detect = +sum([1-(iou[:,k]>=0.5).any().item() for k in range(iou.shape[1])])
                        num_tp_products += true_detect_precision
                        num_rc_products += true_detect_recall
                        num_products += iou.shape[0]
                        num_det += iou.shape[1]
                        num_false += false_detect
                        precision = true_detect_precision/float(iou.shape[1])
                        recall = true_detect_recall/float(iou.shape[0])

# =================================================== detect products in retail boxes @ precision and recall ====================================================== #
                        # if len(within_boxes) and len(retail_boxes):
                        #     # print(within_boxes.shape, retail_boxes.shape)
                        #     iou_retail = box_iou(within_boxes, retail_boxes)
                        #     true_retail_precision = sum([(iou_retail[:,k]>=0.5).any().item() for k in range(iou_retail.shape[1])])
                        #     true_retail_recall = sum([(iou_retail[k]>=0.5).any().item() for k in range(iou_retail.shape[0])])
                        #     false_retail = +sum([1-(iou_retail[:,k]>=0.5).any().item() for k in range(iou_retail.shape[1])])
                        #     retail_precision = true_retail_precision/iou_retail.shape[1]
                        #     retail_recall = true_retail_recall/iou_retail.shape[0]
                        #     num_tp_retails += true_retail_precision
                        #     num_rc_retails += true_retail_recall
                        #     num_retail_det += iou_retail.shape[1]
                        #     num_retail_products += iou_retail.shape[0]
                        # elif len(within_boxes):
                        #     num_retail_products += within_boxes.shape[0]
                        #     num_retail_det += 0
                        # print(precision)
                        # print(recall)
                        # print(retail_precision)
                        # print(retail_recall)
    #                     break
    #         break
    # break

                #         writer.writerow((store, aisle, pano, precision*100, recall*100))
                #         store = ''
                #         aisle = ''
                print("This value is present num_tp_products:",num_tp_products)
                print("This value is present num_det :",num_det)
                avg_precision = num_tp_products / num_det
                avg_recall = num_rc_products / num_products
                avg_false = num_false / num_images
                # avg_retail_precision = num_tp_retails / num_retail_det
                # avg_retail_recall = num_rc_retails / num_retail_products
                # avg_precision = avg_precision / num_images
                # avg_recall = avg_recall / num_images

                iter_dict = {'Ckpt':ckpt,
                             'Chunk Size': chunk_size,
                             'Threshold': thresh,
                             'Chunk Overlap': overlap,
                             'Num Imgs': num_images,
                             'Avg Product Precision (%)': avg_precision*100,
                             'Avg Product Recall (%)': avg_recall*100,
                             'Avg fp/image': avg_false
                             }

                with open(args.out, 'a', newline='') as fsd:
                    fsd_writer = csv.DictWriter(fsd, fieldnames=header, delimiter='\t')
                    fsd_writer.writerow(iter_dict)
                # #     # fsd.write("prod_ckpt_{}_chunk_{}_thresh_{}_overlap_{}: ".format(args.ckpt[5:9], chunk_size, thresh, overlap) + "AP: {:4.4f}, AR: {:4.4f}\n".format(avg_precision*100, avg_recall*100))
                pbar.close()
                pp.pprint(iter_dict)
