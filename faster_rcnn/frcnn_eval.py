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


print("hewwo")
labels_dict = {"pedestrian" : 1}
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

              labels.append(int(elements[0]) + 1)
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


model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers  = 2)

model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)


model.to('cuda:1')


if True:
  model_save_name = 'doas_strat.pt'
  path = F"/home/nkombol/neumre_tester/{model_save_name}" 
  model.load_state_dict(torch.load(path))
  print("=> Loading checkpoint")

print("Model loaded")


training_data = CityPersons('train')
valid_data = CityPersons('valid')
test_data = CityPersons('test')

batch_size = 1

stack_function = lambda x : tuple(zip(*x))

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)


def match_boxes(predicted_boxes, gt_boxes, iou_threshold):

    pred_boxes_tensor = torch.tensor(predicted_boxes).cpu()
    gt_boxes_tensor = torch.tensor(gt_boxes).cpu()

    iou_matrix = torchvision.ops.box_iou(pred_boxes_tensor, gt_boxes_tensor)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)
    iou_matched_pairs = []

    matched_pairs = []
    for row_idx, col_idx in zip(row_indices, col_indices):
        iou = iou_matrix[row_idx, col_idx]
        if iou >= iou_threshold:
            matched_pairs.append((row_idx, col_idx))
            iou_matched_pairs.append(iou)

    return matched_pairs, iou_matched_pairs

model.eval()

datasets = [train_dataloader, valid_dataloader, test_dataloader]
print_names = ["Train", "Validation", "Test"]

for data_loader, print_name in zip(datasets, print_names):
  true_predictions = 0
  accuracy = 0
  gt_box_count = 0
  IoU = 0
  model.to('cuda:1')
  counter = 0
  for data, target in data_loader:
          data = list(img.to('cuda:1') for img in data)
          target = [{k: v.to('cuda:1') for k, v in t.items()} for t in target]
          preds = model(data)
          predicted_boxes = preds[0]['boxes']
          gt_boxes = target[0]['boxes']

          gt_box_count += len(gt_boxes)
    
          iou_threshold = 0.5
          matched_pairs, iou_matched_pairs = match_boxes(gt_boxes, predicted_boxes, iou_threshold)

          for (row_idx, col_idx), iou in zip(matched_pairs, iou_matched_pairs):
              #print(f"GT box {row_idx} matched with Predicted box {col_idx}")
              if target[0]['labels'][row_idx] == preds[0]['labels'][col_idx]:
                #print(f"GT box label {target[0]['labels'][row_idx]} matched with Predicted box label {preds[0]['labels'][col_idx]}")
                IoU += iou
                accuracy += 1

  print(print_name + " accuracy is: ", 1.0 *accuracy / gt_box_count)
  print(print_name + " mean IoU is: ", 1.0 *IoU / gt_box_count)
  print()

"""
Train accuracy is:  0.8191311837218065
Train mean IoU is:  tensor(0.6402)

Validation accuracy is:  0.8307050092764379
Validation mean IoU is:  tensor(0.6399)

Test accuracy is:  0.7774798927613941
Test mean IoU is:  tensor(0.5926)
"""