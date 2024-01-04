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
labels_dict = {"pedestrian" : 0}
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

model.eval()
training_data = CityPersons('train')
valid_data = CityPersons('valid')
test_data = CityPersons('test')

batch_size = 1

stack_function = lambda x : tuple(zip(*x))

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn = stack_function)

def draw_boxes_and_save(image, target, save_name):
    img = transforms.ToPILImage()(image)
    draw = ImageDraw.Draw(img)

    for box, label in zip(target["boxes"], target["labels"]):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=2)
        draw.text((box[0], box[1]), f"Label: {label}", fill="red")

    img.save(save_name)

index_to_visualize = 11

image, target = training_data[index_to_visualize]
image = image.to(torch.device("cuda:1"))
preds = model([image])

output_file_name = "example_gt.jpg" 
draw_boxes_and_save(image, preds[0], output_file_name)
draw_boxes_and_save(image, target, "example_frcnn.jpg")
