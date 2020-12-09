# AriaNN

> TL;DR Benchmark private inference and training on standard neural networks using Function Secret Sharing.

### Info

This is the open-source implementation for the [AriaNN framework paper](https://arxiv.org/abs/2006.04593):

> ARIANN: Low-Interaction Privacy-Preserving Deep Learning via Function Secret Sharing
>
> by _Théo Ryffel, Pierre Tholoniat, David Pointcheval, Francis Bach_

## Usage

### Speed Inference

Benchmark private evaluation of Resnet18 on one batch of size 2 with preprocessing.

```
python main.py --model resnet18 --dataset hymenoptera --batch_size 2 --preprocess
```

### Full evaluation

Compute the test accuracy of a pretrained Alexnet on Tiny Imagenet and compare it to its plain text accuracy.

```
python main.py --model alexnet --dataset tiny-imagenet --batch_size 32 --test
```

### Full training

Train a small CNN on MNIST for 15 epochs and compute accuracy achieved.

```
python main.py --model network2 --dataset mnist --train --epochs 15 --lr 0.01
```

### Documentation

```
usage: main.py [-h] [--model MODEL] [--dataset DATASET]
               [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE]
               [--preprocess] [--fp_only] [--test] [--train] [--epochs EPOCHS]
               [--lr LR] [--websockets] [--verbose]
               [--log_interval LOG_INTERVAL] [--pyarrow_info]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to use for inference (network1, network2, lenet,
                        alexnet, vgg16, resnet18)
  --dataset DATASET     dataset to use (mnist, cifar10, hymenoptera, tiny-
                        imagenet)
  --batch_size BATCH_SIZE
                        size of the batch to use
  --test_batch_size TEST_BATCH_SIZE
                        size of the batch to use
  --preprocess          [only for speed test] preprocess data or not
  --fp_only             Don't secret share values, just convert them to fix
                        precision
  --test                run testing on the complete test dataset
  --train               run training for n epochs
  --epochs EPOCHS       [needs --train] number of epochs to train on
  --lr LR               [needs --train] learning rate of the SGD
  --websockets          use PyGrid nodes instead of a virtual network. (nodes
                        are launched automatically)
  --verbose             show extra information and metrics
  --log_interval LOG_INTERVAL
                        [needs --test or --train] log intermediate metrics
                        every n batches
  --pyarrow_info        print information about PyArrow usage and failure
```

## Installation

### PySyft

Download PySyft from GitHub using the `ryffel/ariaNN` branch.

Install the  dependencies
```
pip install -r pip-dep/requirements.txt
```

Install in editable mode:
```
pip install -e .
```

### PyGrid (for websockets)
Download PyGrid from GitHub using the `ryffel/ariaNN` branch..

First install poetry 
``` 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
take a fresh tab and run `poetry install` in apps/node

### Experiments setup

Write your PyGrid directory in the ``scripts/launch_***.sh``

Define a `HOME` in data.py

You need to install 2 datasets in your `HOME`:
- Tiny Imagenet, from https://github.com/tjmoon0104/pytorch-tiny-imagenet
- Hymenoptera using the instructions:
    ```
    wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
    unzip hymenoptera_data.zip
    ```
    
The working directory is: `cd examples/ariann`
    
    
### Troubleshooting with websockets

If the workers launched automatically complain about syft compression codes,
delete the syft library from the poetry virtual env, active it and install syft 
in editable mode from your GitHub clone

Example of activating the poetry virtual env:
```
source /home/ubuntu/.cache/pypoetry/virtualenvs/openmined.gridnode-L_C_JhA9-py3.6/bin/activate
```

When you first launch the workers, you might have a SQL error. That's not important, just re-run the experiment and it will be gone.

Also, we recommend always testing an experiment locally before using websockets, to make sure everything runs fine.

## Datasets

### MNIST

1 x 28 x 28 pixel images

Suitability: Network1, Network2, LeNet

### CIFAR10

3 x 32 x32 pixel images

Suitability: AlexNet and VGG16

### Tiny Imagenet

3 x 64 x 64 pixel images

Suitability: AlexNet and VGG16

### Hymenoptera

3 x 224 x 224 pixel images

Suitability: ResNet18

## Support the project!

If you're enthusiastic about our project, ⭐️ it to show your support! :heart: