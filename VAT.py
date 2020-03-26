import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAT(object):
  def __init__(self, device, eps, xi, k=1, use_entmin=False):
    self.device = device
    self.xi = xi
    self.eps = eps
    self.k = k
    # self.kl_div = nn.KLDivLoss(size_average=False, reduce=False).to(device)
    self.kl_div = nn.KLDivLoss(reduction='batchmean').to(device)
    self.use_entmin = use_entmin

  def __call__(self, model, X):
      logits = model(X)
      prob_logits = F.softmax(logits.detach(), dim=1)
      d = self.l2_normalize(torch.randn(X.size())) #Generate random unit vector d, using an iid Gaussian distribution
      d = d.to(self.device)

      for ip in range(self.k):
          X_hat = X + d * self.xi
          X_hat.requires_grad = True
          logits_hat = model(X_hat)
          prob_logits_hat = F.log_softmax(logits_hat, dim=1)

          # adv_distance = torch.mean(self.kl_div(prob_logits_hat, prob_logits).sum(dim=1))
          adv_distance = self.kl_div(prob_logits_hat, prob_logits)
          adv_distance.backward()

          d = self.l2_normalize(X_hat.grad).to(self.device)
          model.zero_grad()

      r_adv = d * self.eps
      logits_hat = model(X + r_adv)
      logp_hat = F.log_softmax(logits_hat, dim=1)

      # LDS = torch.mean(self.kl_div(logp_hat, prob_logits).sum(dim=1))
      LDS = self.kl_div(logp_hat, prob_logits)

      if self.use_entmin:
          LDS += self.entropy(logits_hat)

      return LDS

  def l2_normalize(self, d):
      d = d.cpu().numpy()
      d /= (np.sqrt(np.sum(d**2, axis=(1,2,3))).reshape((-1,1,1,1)) + 1e-16)

      return torch.from_numpy(d)

  def entropy(self, logits):
      p = F.softmax(logits, dim=1)
      return -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))
