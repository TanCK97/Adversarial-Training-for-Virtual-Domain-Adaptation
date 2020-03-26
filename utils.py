import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import _is_pil_image, Image

def loadsave(model, optimizer, name, Root, mode='save'):
  assert isinstance(model, torch.nn.Module)
  assert isinstance(name, str)

  string = name + ".pt"
  Save_Path = os.path.join(Root, string)

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

def loadsave(model, name, root, mode='save'):
  assert isinstance(model, torch.nn.Module)
  assert isinstance(name, str)

  string = name + ".pt"
  Save_Path = os.path.join(root, string)

  if (mode == 'save'):
    print("Saving Model")
    torch.save({
        'epoch':                 epoch,
        'model_state_dict':      model.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
    }, Save_Path)

  elif (mode == 'load'):
    print("Loading Model")
    check_point = torch.load(Save_Path)
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
