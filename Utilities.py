import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory for data:', default = './flowers')
parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Architecture type vgg16.')
parser.add_argument('--save_dir', dest= 'save_dir', type = str, default = './checkpoint.pth', help = 'Folder where the model is saved: default is current.')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Gradient descent learning rate')
parser.add_argument('--hidden_layer', type = int, action= 'store', dest = 'hidden_layer', default = 25088, help = 'Number of hidden units #1 for classifier.')

parser.add_argument('--hidden_layer2', type = int, action= 'store', dest = 'hidden_layer2', default = 4096, help = 'Number of hidden units #2 for classifier.')

parser.add_argument('--output_layer', type = int, action= 'store', dest = 'output_layer', default = 102, help = 'Number of output units for classifier.')

parser.add_argument('--epochs', type = int, help = 'Number of epochs', default = 10)

parser.add_argument('--image_path', type=str, help='path of image to be predicted')

parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(args.hidden_layer, args.hidden_layer2)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(0.5)),
                                        ('fc2', nn.Linear(args.hidden_layer2, args.output_layer)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
           
    return model, checkpoint['class_to_idx']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
      
    image = Image.open(image)
        
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    image = preprocess(image)
    
    return image

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    img = process_image(image_path)
    model.to(device)
    img = img.to(device)
    
    img_classes_dict = {v: k for k, v in class_to_idx.items()}
    
    model.eval()
    
    with torch.no_grad():
        img.unsqueeze_(0)
        output = model.forward(img)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for c in classes:
            return_classes.append(img_classes_dict[c])
            
        return probs, return_classes
