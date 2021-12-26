# Imports here
import pickle 
import torch
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import json
import argparse

#get our command line arguments
def get_input_args():
    parser = argparse.ArgumentParser(description='Process values for arguments')
    parser.add_argument('--dir', type = str, default = 'flowers/', help='Path to folder of images')
    parser.add_argument('--arch', type = str, default = 'vgg', help = "Architecture for the NN model")
    parser.add_argument('--hidden_units', type = int, default = 512, help = "Number of hidden training units")
    parser.add_argument('--learnrate', type = float, default = 0.01, help = "Scalar to multiply loss by")
    parser.add_argument('--epochs', type = int, default = 20, help = "Number of rounds of training")
    #parser.add_argument('--flower_cat', type = str, default = 'cat_to_name.json', help = "file with flower categories")
    parser.add_argument('--gpu', type = str, default = 'cuda', help = "use cpu or cuda?")
    return parser.parse_args()

args = get_input_args()

print("dir: ", args.dir)
print("arch: ", args.arch)
print("hidden units: ", args.hidden_units)
print("learnrate: ", args.learnrate)
print("epochs: ", args.epochs)
#print("flower categories file: ", args.flower_cat)
print("device: ", args.gpu)

print("Setting up directories, transforms, and dataloaders...")
data_dir = args.dir
train_dir = data_dir + 'train'
valid_dir = data_dir + 'valid'
test_dir = data_dir + 'test'

#Dataset transforms
train_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#Datasets
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(train_dir, transform=valid_transforms)

#Dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

print("Setting up device")
device = torch.device(args.gpu)

print("Setting up models")
my_models = {"vgg":models.vgg11(pretrained = True), "densenet":models.densenet121(pretrained = True)}
model = my_models[args.arch]#models.densenet121(pretrained = True)#my_models[args.arch]#models.vgg11(pretrained = True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

print("Setting up classifiers")
my_classifiers = {}
my_classifiers["vgg"] = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))
my_classifiers["densenet"] = nn.Sequential(nn.Linear(1024, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(args.hidden_units, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
model.classifier = my_classifiers[args.arch]
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learnrate)

model.to(device);


print("Model Prepared?")
print(model)

# TODO: Do validation on the test set

train_losses, test_losses = [], []
epochs = args.epochs
steps = 0

print("Training and Testing Model ", epochs, " times")
for e in range(epochs):
    running_loss = 0
    print("Training...")
    for images, labels in train_dataloader:
        steps += 1
        # Move input and label tensors to the default device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(images)
        loss = criterion(logps, labels)
        loss.backward()
        print("Epoch: ", e + 1, "Training Loss: ", loss.item())
        optimizer.step()
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        print("Testing...")
        ## TODO: Implement the validation pass and print out the validation accuracy
        with torch.no_grad():
            model.eval()
            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        model.train()
        train_losses.append(running_loss/len(train_dataloader))
        test_losses.append(test_loss/len(valid_dataloader))
        print("Epoch:  {}/{}.. ".format(e+1, epochs),
              "Train Loss: {:.3f}.. ".format(running_loss/len(train_dataloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(valid_dataloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(valid_dataloader)))

#Save the checkpoint
print("Saving checkpoint file")
checkpoint = {'arch' : args.arch,
              'classifier' : model.classifier,
              'epochs': args.epochs,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'device' : device}

torch.save(checkpoint.state_dict(), 'checkpoint.pth')
print("Finished Training file")
