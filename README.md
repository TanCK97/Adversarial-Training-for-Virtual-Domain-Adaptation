# Adversarial Training for Virtual Domain Adaptation
This repository contains an implementation of LeNet and Virtual Adversarial Training (VAT) for domain adaptation. Currently only supports SVHN to MNIST

## Dependencies
* numpy
* matplotlib
* os
* argparse
* torch
* torchvision
* sklearn

## Training model
```bash
python3 train.py \
        --lr 0.0001 \
        --epochs 20 \
        --alpha 1.0 \
        --eps 4.5 \
        --xi 10.0 \
        --k 1 \
        --use-entmin \
        --weights-path ./weights \
        --dataset-path ./dataset \
```

## Testing model
```bash
python3 train.py \
        --weights-path ./weights \
        --dataset-path ./dataset \
        --eval-only
```
