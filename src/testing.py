
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import time
import argparse
import wandb
from torch.utils.data import Dataset, DataLoader
import glob

from data.dataset import get_data_loader
from utils import load_brains, set_seed, crop, init_cuda, init_fabric, init_wandb
from models.metrics import Dice
from models.segformer import Segformer
from training.trainer import Trainer

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('--wandb_description', help="Description add to the wandb run", type=str,required=False)

args = parser.parse_args()

print(args.wandb_description is None)

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# from lightning.fabric import Fabric

# # Device configuration
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# fabric = Fabric(devices=2,num_nodes=1,strategy='dp')
# fabric.launch()

# # Hyper-parameters
# input_size = 784 # 28x28
# hidden_size = 500
# num_classes = 10
# num_epochs = 1
# batch_size = 1
# learning_rate = 0.001

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='./data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

# # visualize images
# # for i in range(6):
# #     plt.subplot(2,3,i+1)
# #     plt.imshow(train_dataset[i][0][0], cmap='gray')
# # plt.show()


# # Fully connected neural network with one hidden layer
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.l1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out

# model = NeuralNet(input_size, hidden_size, num_classes)#.to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model,optimizer = fabric.setup(model,optimizer)
# train_loader,test_loader = fabric.setup_dataloaders(train_loader,test_loader)

# # Train the model
# n_total_steps = len(train_loader)
# print(n_total_steps)
# batch_idx = 0
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         if batch_idx == 10:
#             break
#         # origin shape: [100, 1, 28, 28]
#         # resized: [100, 784]
#         # reshape to match dimension of first linear layer
#         images = images.reshape(-1, 28*28)#.to(device)
#         print(f'image {batch_idx}:', torch.sum(images))
#         batch_idx += 1
#         labels = labels#.to(device)
#         # images = fabric.to_device(images.reshape(-1, 28*28))#.to(device)
#         # labels = fabric.to_device(labels)#.to(device)

#         # Forward pass and loss calculation
#         # [100, 10]
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         # Backward and optimize
#         # loss.backward()
#         fabric.backward(loss)
#         optimizer.step()
#         optimizer.zero_grad()

#         # if (i+1) % 100 == 0:
#         #     print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# # # Test the model: we don't need to compute gradients
# # with torch.no_grad():
# #     n_correct = 0
# #     n_samples = len(test_loader.dataset)

# #     for images, labels in test_loader:
# #         # [100, 784]
# #         images = images.reshape(-1, 28*28)#.to(device)
# #         # [100]
# #         labels = labels#.to(device)

# #         # # [100, 784]
# #         # images = fabric.to_device(images.reshape(-1, 28*28))#.to(device)
# #         # # [100]
# #         # labels = fabric.to_device(labels)#.to(device)

# #         # [100, 10]
# #         outputs = model(images)

# #         # max returns (output_value ,index)
# #         # [100]
# #         _, predicted = torch.max(outputs, 1)
# #         # print(predicted == labels)
# #         n_correct += (predicted == labels).sum().item()

# #     acc = n_correct / n_samples
# #     print(f'Accuracy of the network on the {n_samples} test images: {100*acc} %')