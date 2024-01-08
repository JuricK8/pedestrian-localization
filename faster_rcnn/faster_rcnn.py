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
import sys
import datetime
import os

output_file = open("log.txt", "a")

sys.stdout = output_file


print("hewwo", flush=True)
labels_dict = {"pedestrian" : 0}
class CityPersons(Dataset):
    def __init__(self, type):
      self.img_dir = "/home/nkombol/neumre_tester/citypersons/" + type + "/images"  
      self.label_dir = "/home/nkombol/neumre_tester/citypersons/" + type + "/annotations"

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

              labels.append(int(elements[0])) #+ 1) SAMO KAD RADIŠ S JEDNOM KLASOM 
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


model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT', trainable_backbone_layers = 3) 

model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 6) # = 2) KAD SAMO PEDESTRIAN

model.to('cuda:0')
print("Model loaded", flush=True)

train_again = False
train_on_gpu = torch.cuda.is_available()


training_data = CityPersons('train')
valid_data = CityPersons('valid')
test_data = CityPersons('test')

batch_size = 10

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

num_epochs = 30
model.to("cuda:0")

non_frozen_parameters = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(non_frozen_parameters, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

if False:
  model_save_name = 'doas_strat18.pt'
  path = F"/home/nkombol/neumre_tester/{model_save_name}" 
  model.load_state_dict(torch.load(path))

  optimizer.load_state_dict(torch.load("/home/nkombol/neumre_tester/doas_strat18_optim.pt"))
  print("=> Loading checkpoint")
  for i in range(19):
    lr_scheduler.step()

n_epochs = [*range(num_epochs)]

train_losslist = []
valid_loss_min = np.Inf 

for epoch in n_epochs:
  train_loss = 0.0
      
  counter = 0
  model.train()
  losses_print = {"loss_classifier" : 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
  for data, target in train_dataloader: 
    optimizer.zero_grad()
    data = list(img.cuda() for img in data)
    target = [{k : v.cuda() for k, v in t.items()} for t in target]

    
    loss_dict = model(data, target)

    losses_print["loss_classifier"] += loss_dict["loss_classifier"].detach()
    losses_print["loss_box_reg"] += loss_dict["loss_box_reg"].detach() 
    losses_print["loss_objectness"] += loss_dict["loss_objectness"].detach() 
    losses_print["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].detach() 
    losses = sum(loss for loss in loss_dict.values())
            
    losses.backward()
    optimizer.step()
    train_loss += losses
    if counter % 960 == 0:
      print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "epoch: {}, counter: {}, batch_loss: {}".format(epoch + 0, counter, losses), flush=True)
    counter +=batch_size

  print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), losses_print, flush=True)
  lr_scheduler.step()
  valid_loss = 0.0
  losses_print = {"loss_classifier" : 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
        
  with torch.no_grad():
    for data, target in valid_dataloader:
      optimizer.zero_grad()
      data = list(img.cuda() for img in data)
      target = [{k : v.cuda() for k, v in t.items()} for t in target]
      loss_dict = model(data, target)
      losses = sum(loss for loss in loss_dict.values())

      losses_print["loss_classifier"] += loss_dict["loss_classifier"].detach() 
      losses_print["loss_box_reg"] += loss_dict["loss_box_reg"].detach() 
      losses_print["loss_objectness"] += loss_dict["loss_objectness"].detach() 
      losses_print["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].detach() 
            
      valid_loss += losses
              
  train_loss = train_loss
  valid_loss = valid_loss

  train_losslist.append(train_loss)
            
  print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), losses_print, flush=True)
  print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch + 0, train_loss, valid_loss), flush=True)
        
  if valid_loss <= valid_loss_min or True: #OVO JE JERE SAM LJENA
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss), flush=True)
    model_save_name = 'doas_strat' + str(epoch) +'.pt'
    path = F"/home/nkombol/neumre_tester/{model_save_name}" 
    torch.save(model.state_dict(), path)

    optimizer_save_name = 'doas_strat'+ str(epoch) +'_optim.pt'
    path = F"/home/nkombol/neumre_tester/{optimizer_save_name}" 
    torch.save(optimizer.state_dict(), path)

    valid_loss_min = valid_loss

sys.stdout = sys.__stdout__

output_file.close()


