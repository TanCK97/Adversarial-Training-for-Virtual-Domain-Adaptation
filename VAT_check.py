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

from utils import toRGB, loadsave
from LeNet import LeNet
from VAT import VAT

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

def barchartplot(classifier, loader, device):
    assert isinstance(classifier, torch.nn.Module)
    assert isinstance(loader, torch.utils.data.DataLoader)
    assert isinstance(device, torch.device)

    classifier.eval()

            # expected output
            #0,1,2,3,4,5,6,7,8,9
    data = [[0,0,0,0,0,0,0,0,0,0], # 0
            [0,0,0,0,0,0,0,0,0,0], # 1
            [0,0,0,0,0,0,0,0,0,0], # 2
            [0,0,0,0,0,0,0,0,0,0], # 3
            [0,0,0,0,0,0,0,0,0,0], # 4
            [0,0,0,0,0,0,0,0,0,0], # 5
            [0,0,0,0,0,0,0,0,0,0], # 6
            [0,0,0,0,0,0,0,0,0,0], # 7
            [0,0,0,0,0,0,0,0,0,0], # 8
            [0,0,0,0,0,0,0,0,0,0]] # 9  model output

    with torch.no_grad():
        for x, y in loader:
            prob_y = F.softmax(classifier(x.to(device)), dim=1)
            pred_y = torch.max(prob_y, dim=1)[1]
            pred_y = pred_y.to(torch.device('cpu'))
            data[pred_y][y] += 1

    X = np.arange(10)
    ax = plt.subplot(111)

    ax.bar(X + 0.00, data[0], color='red', width=0.1, align='edge')
    ax.bar(X + 0.10, data[1], color='blue', width=0.1, align='edge')
    ax.bar(X + 0.20, data[2], color='green', width=0.1, align='edge')
    ax.bar(X + 0.30, data[3], color='yellow', width=0.1, align='edge')
    ax.bar(X + 0.40, data[4], color='purple', width=0.1, align='edge')
    ax.bar(X + 0.50, data[5], color='violet', width=0.1, align='edge')
    ax.bar(X + 0.60, data[6], color='gray', width=0.1, align='edge')
    ax.bar(X + 0.70, data[7], color='brown', width=0.1, align='edge')
    ax.bar(X + 0.80, data[8], color='pink', width=0.1, align='edge')
    ax.bar(X + 0.90, data[9], color='cyan', width=0.1, align='edge')

    plt.legend(['0','1','2','3','4','5','6','7','8','9'])
    plt.xticks(np.arange(10), ['0','1','2','3','4','5','6','7','8','9'])
    plt.show()

def train(model, optimizer, criterion, criterion_VAT, train_labelled_loader, train_unlabelled_loader, valloader, testloader, alpha, epochs, device, root):
    best_acc = 0.0
    supervised_loss = []
    unsupervised_loss = []

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.5)

    for epoch in range(epochs):  # loop over the dataset multiple times
        dataloader_iterator = iter(train_unlabelled_loader)

        for i, data in enumerate(train_labelled_loader):
            try:
                data2 = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(train_unlabelled_loader)
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

        scheduler.step()

            # Calculating loss and accuracy
        vat_acc =  evaluate_classifier(model, valloader, device)
        print('Epoch: {}, Val_acc: {:.3} Sup_loss: {:.3} Unsup_loss: {:.3}'.format(epoch, vat_acc, sup_loss.item(), unsup_loss.item()))

        supervised_loss.append(sup_loss.item())
        unsupervised_loss.append(unsup_loss.item())

        if (vat_acc > best_acc):
          loadsave(model, optimizer, "VATcheck", root=root, mode='save')
          best_acc = vat_acc

    return supervised_loss, unsupervised_loss

def main(args):
    transform_SVHN = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='train', download=True, transform=transform_SVHN)
    testset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='test', download=True, transform=transform_SVHN)

    train_labelled_size = int(0.6 * len(trainset_SVHN))
    train_unlabelled_size = len(trainset_SVHN) - train_labelled_size
    val_size = int(0.2 * len(testset_SVHN))
    test_size = len(testset_SVHN) - val_size
    trainset_labelled, trainset_unlabelled = torch.utils.data.random_split(trainset_SVHN, [train_labelled_size, train_unlabelled_size])
    valset, testset = torch.utils.data.random_split(testset_SVHN, [val_size, test_size])

    # Should increase batch size to decrease training time. Batch size for LeNet and VAT datasets can be different, i.e. 32 for LeNet and 128 for VAT
    train_labelled_loader = DataLoader(trainset_labelled, batch_size=32, shuffle=True, num_workers=2)
    train_unlabelled_loader = DataLoader(trainset_unlabelled, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

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
        loadsave(lenet0, optimizer, "VATcheck", root=args.weights_path[0], mode='load')

    else:
        supervised_loss, unsupervised_loss = train(lenet0, optimizer, criterion, criterion_VAT, train_labelled_loader, train_unlabelled_loader, valloader, testloader, args.alpha, args.epochs, device, args.weights_path[0])

        plt.subplot(2,1,1)
        plt.plot(supervised_loss)
        plt.title("Supervised loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(unsupervised_loss)
        plt.title("Unsupervised loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.show()

        loadsave(lenet0, optimizer, "VATcheck", root=args.weights_path[0], mode='load')

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
