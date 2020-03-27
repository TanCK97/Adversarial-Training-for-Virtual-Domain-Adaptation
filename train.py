import argparse
import os

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

  classifier.train()

  return correct/total

def train(model, optimizer, criterion, criterion_VAT, trainloader_SVHN, trainloader_MNIST, valloader, alpha, epochs, device, root):
    Save_Path = os.path.join(root, "LenetVAT.pt")
    best_acc = 0.0

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
      vacc =  evaluate_classifier(model, valloader, device)
      print('Epoch: {}, Val_acc: {:.3} Sup_loss: {:.3} Unsup_loss: {:.3}'.format(epoch, vacc, sup_loss.item(), unsup_loss.item()))

      if (vacc > best_acc):
        print("Saving Model from Epoch %d" %epoch)
        torch.save({
            'epoch':                 epoch,
            'model_state_dict':      model.state_dict(),
            'optimizer_state_dict':  optimizer.state_dict(),
        }, Save_Path)
        best_acc = vacc

def test(model, testloader, device):
    acc = evaluate_classifier(model, testloader, device)
    print('Accuracy of the network is %d%%\n' %(100*acc))

def main(args):
    transform_SVHN = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_MNIST = transforms.Compose([toRGB(), transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset_SVHN = torchvision.datasets.SVHN(root=args.dataset_path[0], split='train', download=True, transform=transform_SVHN)
    fullset_MNIST = torchvision.datasets.MNIST(root=args.dataset_path[0], train=True, download=True, transform=transform_MNIST)
    testset = torchvision.datasets.MNIST(root=args.dataset_path[0], train=False, download=True, transform=transform_MNIST)

    train_size = int(0.8 * len(fullset_MNIST))
    val_size = len(fullset_MNIST) - train_size
    trainset_MNIST, valset = torch.utils.data.random_split(fullset_MNIST, [train_size, val_size])

    trainloader_SVHN = DataLoader(trainset_SVHN, batch_size=5, shuffle=True, num_workers=2)
    trainloader_MNIST = DataLoader(trainset_MNIST, batch_size=5, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    lenet0 = LeNet(device)
    lenet0 = lenet0.to(device)
    print(lenet0)

    criterion = nn.CrossEntropyLoss()
    criterion_VAT = VAT(device, eps=args.eps, xi=args.xi, use_entmin=args.use_entmin)
    optimizer = optim.Adam(lenet0.parameters(), lr=args.lr)

    if args.eval_only:
        loadsave(lenet0, optimizer, args.weights_path[0], mode='load')
        acc =  evaluate_classifier(lenet0, testloader, device)
        print("Accuracy of the network is %d%%\n" %(acc*100))
    else:
        train(lenet0, optimizer, criterion, criterion_VAT, trainloader_SVHN, trainloader_MNIST, valloader, args.alpha, args.epochs, device, args.weights_path[0])
        acc =  evaluate_classifier(lenet0, testloader, device)
        print("Accuracy of the network is %d%%\n" %(acc*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to train the Lenet-VAT network"
    )
    parser.add_argument(
        "--lr",
        default=0.0001,
        nargs='?',
        help="Learning rate",
        type=float
    )

    parser.add_argument(
        "--epochs",
        default=20,
        nargs='?',
        help="Number of epochs",
        type=int
    )

    parser.add_argument(
        "--alpha",
        default = 2.0,
        nargs='?',
        help="Alpha value",
        type=float
    )

    parser.add_argument(
        "--eps",
        default=4.5,
        nargs='?',
        help="Epsilon value",
        type=float
    )

    parser.add_argument(
        "--xi",
        default=100.0,
        nargs='?',
        help="xi value",
        type=float
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
