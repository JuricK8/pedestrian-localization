import numpy as np 
import matplotlib.pyplot as plt
import os
import PIL
import torch
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import Dataset,DataLoader
from scipy.optimize import linear_sum_assignment
import torchvision
import torch.optim as optim
import cv2
import matplotlib.patches as patches

from PIL import Image, ImageDraw
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

import sys
import datetime

class CityPersons(Dataset):
    def __init__(self, type):
      self.img_dir = "/home/nkombol/neumre_tester/citypersons/" + type + "/images"  
      self.label_dir = "/home/nkombol/neumre_tester/citypersons/" + type + "/labels"

      self.images = []
      for filename in os.listdir(self.label_dir):
        filepath = os.path.join(self.label_dir, filename)

        if os.path.isfile(filepath) and filename.endswith('.txt'):

            if os.path.getsize(filepath) != 0:
                self.images.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx][:-3] + "jpg")

        transform = transforms.ToTensor()
        image = transform(PIL.Image.open(img_path).convert("RGB"))

        label_path = os.path.join(self.label_dir, self.images[idx][:-3] + "txt")

        boxes = []
        labels = []

        with open(label_path, 'r') as file:
          for annot in file:
              elements = annot.strip().split()

              labels.append(int(elements[0]))
              #1280 x 640
              xmin, ymin, width, height = elements[1:]
              
              xmin, ymin, width, height = [float(xmin) * 1280, float(ymin) * 640, float(width) * 1280, float(height) * 640]
              xmin, ymin, width, height = [xmin - width/2, ymin - height/2, xmin + width/2, ymin + height/2]
              boxes.append([xmin, ymin, width, height])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float)
        labels = torch.as_tensor(labels, dtype=torch.int64 )
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        return image, target



model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers  = 3)

model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 6)  #2)

model.to('cuda:1')


training_data = CityPersons('train')
valid_data = CityPersons('valid')
test_data = CityPersons('test')

batch_size = 1

stack_function = lambda x : tuple(zip(*x))

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)


model.eval()

datasets = [valid_dataloader] #[train_dataloader, valid_dataloader, test_dataloader]
print_names = ["Validation"] #["Train", "Validation", "Test"]


def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5):

    def calculate_iou(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)

        return iou

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

    true_positives = 0
    false_positives = 0
    num_gt_boxes = len(gt_boxes)
    precision_at_recall = []

    for pred_box in pred_boxes:
        iou_max = 0
        match_index = -1

        for i, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(gt_box, pred_box[:4])
            if iou > iou_max:
                iou_max = iou
                match_index = i

        if iou_max >= iou_threshold:
            if gt_boxes[match_index][4] == 0:  
                true_positives += 1
                gt_boxes[match_index][4] = 1  
            else:
                false_positives += 1
        else:
            false_positives += 1

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / num_gt_boxes
        precision_at_recall.append((recall, precision))

    average_precision = 0.0
    previous_recall = 0.0

    for recall, precision in sorted(precision_at_recall, key=lambda x: x[0]):
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    for cnt, box in enumerate(gt_boxes):
        gt_boxes[cnt][4] = 0
    return average_precision



def calculate_map50_95(gt_boxes, pred_boxes):
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)] 
    map_sum = 0.0

    for iou_threshold in iou_thresholds:
        map_at_iou = calculate_ap(gt_boxes, pred_boxes, iou_threshold)
        map_sum += map_at_iou

    map50_95 = map_sum / len(iou_thresholds)
    return map50_95

for i in range(30):
    model_save_name = 'doas_strat' +str(i)+ '.pt'
    path = F"/home/nkombol/neumre_tester/5class 2 unforzen/{model_save_name}" 
    model.load_state_dict(torch.load(path))
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), model_save_name," Loading checkpoint")
    
    for data_loader, print_name in zip(datasets, print_names):
        model.to('cuda:1')
        counter = 0
        ap_50 = 0
        ap_50_95 = 0
        for data, target in data_loader:
                data = list(img.to('cuda:1') for img in data)
                target = [{k: v.to('cuda:1') for k, v in t.items()} for t in target]
                preds = model(data)
                predicted_boxes = []
                for cnt, p in enumerate(preds[0]['boxes']):
                    predicted_boxes.append(torch.cat((p,torch.tensor([preds[0]['scores'][cnt]]).to('cuda:1')), 0))
                gt_boxes = []
                for b in target[0]['boxes']:
                    gt_boxes.append(torch.cat((b, torch.tensor([0]).to('cuda:1')), 0))

                ap_50 += calculate_ap(gt_boxes.copy(), predicted_boxes.copy(), iou_threshold=0.5)
                
                ap_50_95 += calculate_map50_95(gt_boxes, predicted_boxes)
                
                counter += 1
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),print_name, "maAP50 : ", ap_50*1.0/counter)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),print_name, "mAP50-95: ", ap_50_95*1.0/counter)

