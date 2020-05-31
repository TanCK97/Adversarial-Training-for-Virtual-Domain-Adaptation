import argparse
import os
import inspect
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils import toRGB, loadsave, evaluate_classifier, barchartplot, loss_plot
from LeNet import LeNet

def evaluate_classifier(classifier, loader, device):
  assert isinstance(classifier, torch.nn.Module)
  assert isinstance(loader, torch.utils.data.DataLoader)
  assert isinstance(device, torch.device)

  classifier.eval()

  # n_err = 0
  correct = 0
  total = 0
  with torch.no_grad():
      for x, y in loader:
          prob_y = F.softmax(classifier(x.to(device)), dim=1)
          pred_y = torch.max(prob_y, dim=1)[1]
          pred_y = pred_y.to(torch.device('cpu'))
          correct += (pred_y == y).sum().item()
          total += y.size(0)
  vat_acc = correct/total

  classifier.train()

  return vat_acc

def train(model, optimizer, criterion, trainloader, valloader, testloader, epochs, device, root):
    best_acc = 0.0
    supervised_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
      for i, data in enumerate(trainloader):

        l_x, l_y = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(l_x)
        sup_loss = criterion(outputs, l_y)
        loss = sup_loss

        loss.backward()
        optimizer.step()

      # Calculating loss and accuracy
      vat_acc =  evaluate_classifier(model, valloader, device)
      print('Epoch: {}, Val_acc: {:.3} Sup_loss: {:.3}'.format(epoch, vat_acc, sup_loss.item()))

      supervised_loss.append(sup_loss.item())

      if (vat_acc > best_acc):
        loadsave(model, optimizer, "Lenet", root=root, mode='save')
        best_acc = vat_acc

    return supervised_loss

def main(args):
    transform_SVHN = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_MNIST = transforms.Compose([toRGB(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='train', download=True, transform=transform_SVHN)
    testset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='test', download=True, transform=transform_SVHN)
    testset_MNIST = torchvision.datasets.MNIST(root=args.dataset_path[0], train=False, download=True, transform=transform_MNIST)

    discard_size = int(0.4 * len(trainset_SVHN))
    train_size = len(trainset_SVHN) - discard_size
    val_size = int(0.2 * len(testset_MNIST))
    test_size = len(testset_MNIST) - val_size
    trainset, discardset = torch.utils.data.random_split(trainset_SVHN, [train_size, discard_size])
    valset, testset = torch.utils.data.random_split(testset_MNIST, [val_size, test_size])

    # Should increase batch size to decrease training time. Batch size for LeNet and VAT datasets can be different, i.e. 32 for LeNet and 128 for VAT
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    lenet0 = LeNet(device)
    lenet0 = lenet0.to(device)
    print(lenet0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet0.parameters(), lr=args.lr) # Should implement lr scheduler.
    # optimizer = optim.SGD(lenet0.parameters(), lr=args.lr, momentum=0.9)

    if args.eval_only:
        loadsave(lenet0, optimizer, "Lenet", root=args.weights_path[0], mode='load')

    else:
        supervised_loss = train(lenet0, optimizer, criterion, trainloader, valloader, testloader, args.epochs, device, args.weights_path[0])

        plt.subplot(1,1,1)
        plt.plot(supervised_loss)
        plt.title("Supervised loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.show()

        loadsave(lenet0, optimizer, "Lenet", root=args.weights_path[0], mode='load')

    vat_acc =  evaluate_classifier(lenet0, testloader, device)
    print("Accuracy of the network on SVHN is %d%%\n" %(vat_acc*100))

    barchartplot(lenet0, testloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train the Lenet-VAT network"
    )
    parser.add_argument(
        "--lr",
        default=0.0001, # Original VAT paper used ADAM with lr = 0.001
        nargs='?',
        help="Learning rate",
        type=float
    )

    parser.add_argument(
        "--epochs",
        default=20, # Might have to increase, VAT paper used 84 to train SVHN
        nargs='?',
        help="Number of epochs",
        type=int
    )

    parser.add_argument(
        "--weights-path",
        default=['./weights'],
        nargs="+",
        help="Path to the weights",
        type=str
    )

    parser.add_argument(
        "--dataset-path",
        default=['./dataset'],
        nargs="+",
        help="Path to the dataset",
        type=str
    )

    parser.add_argument(
        "--eval-only",
        help="Set model to evaluation mode",
        action='store_true'
    )

    args = parser.parse_args()

    main(args)
