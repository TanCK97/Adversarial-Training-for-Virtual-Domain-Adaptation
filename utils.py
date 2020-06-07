import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import _is_pil_image, Image
import torch.nn.functional as F

from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

def loadsave(model, optimizer, name, root, mode='save'):
  assert isinstance(model, torch.nn.Module)
  assert isinstance(name, str)

  string = name + ".pt"
  Save_Path = os.path.join(root, string)

  if (mode == 'save'):
    print("Saving Model")
    torch.save({
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
    }, Save_Path)

  elif (mode == 'load'):
    print("Loading Model")
    check_point = torch.load(Save_Path)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])

class toRGB(object):
  def __call__(self, img):
    """Convert image to rgb version of image.
    Args:
          img (PIL Image): Image to be converted to grayscale.
    Returns:
        PIL Image
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img, np_img, np_img])
    img = Image.fromarray(np_img, 'RGB')
    return img

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def evaluate_classifier(classifier, loader, loader_org, device):
  assert isinstance(classifier, torch.nn.Module)
  assert isinstance(loader, torch.utils.data.DataLoader)
  assert isinstance(loader_org, torch.utils.data.DataLoader)
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

  correct = 0
  total = 0
  with torch.no_grad():
      for x, y in loader_org:
          prob_y = F.softmax(classifier(x.to(device)), dim=1)
          pred_y = torch.max(prob_y, dim=1)[1]
          pred_y = pred_y.to(torch.device('cpu'))
          correct += (pred_y == y).sum().item()
          total += y.size(0)
  org_acc = correct/total

  classifier.train()

  return vat_acc, org_acc

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

    # Calculating the NMI
    num_vec = []
    cls_vec = []

    for cls in range(10):
        for num in range(10):
            num_vec.extend([num]*data[num][cls])
            cls_vec.extend([cls]*data[num][cls])


    nmi = normalized_mutual_info_score(cls_vec, num_vec)
    print("NMI: " + str(nmi))

def loss_plot(supervised_loss, unsupervised_loss):
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
