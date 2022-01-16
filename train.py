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

parser.add_argument('--epochs', type = int, help = 'Number of epochs', default = 1)

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

epochs = args.epochs
steps = 0
print_every = 40
model.to(device)

for e in range(epochs):
    model.train()
    running_loss = 0
    
    for images, labels in iter(trainloader):
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            with torch.no_grad():
                test_loss, accuracy = validation(model, testloader, criterion)
            
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            
            model.train()

model.class_to_idx = train_data.class_to_idx

print("Saving checkpoint file")
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epochs': epochs,
              'model': getattr(models, args.arch)(pretrained=True),
              'classifier': classifier,
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}
for key in checkpoint.keys():
     print(key)
        
torch.save(checkpoint, '/checkpoint.pth')
print("Finished Training file")
