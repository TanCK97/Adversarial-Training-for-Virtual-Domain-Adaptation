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

from utils import toRGB, loadsave, evaluate_classifier, loss_plot, barchartplot
from LeNet import LeNet
from VAT import VAT

def train(model, optimizer, criterion, criterion_VAT, trainloader_SVHN, trainloader_MNIST, valloader, testloader_SVHN, alpha, epochs, device, root):
    best_acc = 0.0
    supervised_loss = []
    unsupervised_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
      dataloader_iterator = iter(trainloader_MNIST)

      for i, data in enumerate(trainloader_SVHN):
        try:
            data2 = next(dataloader_iterator)
        except StopIteration:
          dataloader_iterator = iter(trainloader_MNIST)
          data2 = next(dataloader_iterator)

        l_x, l_y = data[0].to(device), data[1].to(device)
        ul_x, ul_y = data2[0].to(device), data2[1].to(device)
        optimizer.zero_grad()

        outputs = model(l_x)
        sup_loss = criterion(outputs, l_y)
        unsup_loss = alpha * criterion_VAT(model, ul_x)
        loss = sup_loss + unsup_loss

        loss.backward()
        optimizer.step()

      # Calculating loss and accuracy
      vat_acc, org_acc =  evaluate_classifier(model, valloader,testloader_SVHN, device)
      print('Epoch: {}, Val_acc: {:.3} Org_acc: {:.3} Sup_loss: {:.3} Unsup_loss: {:.3}'.format(epoch, vat_acc, org_acc, sup_loss.item(), unsup_loss.item()))

      supervised_loss.append(sup_loss.item())
      unsupervised_loss.append(unsup_loss.item())

      if (vat_acc > best_acc):
        loadsave(model, optimizer, "LenetVAT", root=root, mode='save')
        best_acc = vat_acc

    return supervised_loss, unsupervised_loss

def main(args):
    transform_SVHN = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_MNIST = transforms.Compose([toRGB(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='train', download=True, transform=transform_SVHN)
    fullset_MNIST = torchvision.datasets.MNIST(root=args.dataset_path[0], train=True, download=True, transform=transform_MNIST)
    testset = torchvision.datasets.MNIST(root=args.dataset_path[0], train=False, download=True, transform=transform_MNIST)
    testset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='test', download=True, transform=transform_SVHN)

    train_size = int(0.8 * len(fullset_MNIST))
    val_size = len(fullset_MNIST) - train_size
    trainset_MNIST, valset = torch.utils.data.random_split(fullset_MNIST, [train_size, val_size])

    # Should increase batch size to decrease training time. Batch size for LeNet and VAT datasets can be different, i.e. 32 for LeNet and 128 for VAT
    trainloader_SVHN = DataLoader(trainset_SVHN, batch_size=32, shuffle=True, num_workers=2)
    trainloader_MNIST = DataLoader(trainset_MNIST, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    testloader_SVHN = DataLoader(testset_SVHN, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))

    lenet0 = LeNet(device)
    lenet0 = lenet0.to(device)
    print(lenet0)

    criterion = nn.CrossEntropyLoss()
    criterion_VAT = VAT(device, eps=args.eps, xi=args.xi, k=args.k, use_entmin=args.use_entmin)
    optimizer = optim.Adam(lenet0.parameters(), lr=args.lr) # Should implement lr scheduler.
    # optimizer = optim.SGD(lenet0.parameters(), lr=args.lr, momentum=0.9)

    if args.eval_only:
        loadsave(lenet0, optimizer, "LenetVAT", root=args.weights_path[0], mode='load')

    else:
        supervised_loss, unsupervised_loss = train(lenet0, optimizer, criterion, criterion_VAT, trainloader_SVHN, trainloader_MNIST, valloader, testloader_SVHN, args.alpha, args.epochs, device, args.weights_path[0])
        loss_plot(supervised_loss, unsupervised_loss)
        loadsave(lenet0, optimizer, "LenetVAT", root=args.weights_path[0], mode='load')

        # loadsave(lenet0, optimizer, "LenetVAT", root=args.weights_path[0], mode='load')

    vat_acc, org_acc =  evaluate_classifier(lenet0, testloader, testloader_SVHN, device)
    print("Accuracy of the network on MNIST is %d%%\nAccuracy of the network on SVHN is %d%%\n" %(vat_acc*100, org_acc*100))

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
        "--alpha",
        default = 1.0, # Research shows that for small values of eps changing alpha has little effect, so default is 1.0
        nargs='?',
        help="Alpha value",
        type=float
    )

    parser.add_argument(
        "--eps",
        default=4.5, # Try within range of [0.05, 10.0]
        nargs='?',
        help="Epsilon value",
        type=float
    )

    parser.add_argument(
        "--xi",
        default=10.0, # 100 might be too much, so try original default value of 10.0. Can also try within range [1,10] but only as last resort
        nargs='?',
        help="xi value",
        type=float
    )

    parser.add_argument(
        "--k",
        default=1, # Can increase this to potentially improve results, but training time will linearly increase and results might not change much
        nargs='?',
        help="k value",
        type=int
    )

    parser.add_argument(
        "--use-entmin",
        help="Set model to use conditional entropy as an additional cost",
        action='store_true'
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
