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
usage: main.py [-h] [--model MODEL] [--dataset DATASET] [--batch_size BATCH_SIZE] [--test_batch_size TEST_BATCH_SIZE] [--preprocess] [--fp_only] [--public] [--test] [--train] [--epochs EPOCHS]
               [--lr LR] [--momentum MOMENTUM] [--websockets] [--verbose] [--log_interval LOG_INTERVAL] [--comm_info] [--pyarrow_info]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to use for inference (network1, network2, lenet, alexnet, vgg16, resnet18)
  --dataset DATASET     dataset to use (mnist, cifar10, hymenoptera, tiny-imagenet)
  --batch_size BATCH_SIZE
                        size of the batch to use. Default 128.
  --test_batch_size TEST_BATCH_SIZE
                        size of the batch to use
  --preprocess          [only for speed test] preprocess data or not
  --fp_only             Don't secret share values, just convert them to fix precision
  --public              [needs --train] Train without fix precision or secret sharing
  --test                run testing on the complete test dataset
  --train               run training for n epochs
  --epochs EPOCHS       [needs --train] number of epochs to train on. Default 15.
  --lr LR               [needs --train] learning rate of the SGD. Default 0.01.
  --momentum MOMENTUM   [needs --train] momentum of the SGD. Default 0.9.
  --websockets          use PyGrid nodes instead of a virtual network. (nodes are launched automatically)
  --verbose             show extra information and metrics
  --log_interval LOG_INTERVAL
                        [needs --test or --train] log intermediate metrics every n batches. Default 10.
  --comm_info           Print communication information
  --pyarrow_info        print information about PyArrow usage and failure
```

## Installation

### With Docker

```
docker-compose up
```

Connect to the container:
```
docker exec -ti ariann /bin/bash 
```

You can already start running the first experiments! Try:
``` 
python main.py --model alexnet --dataset cifar10 --batch_size 128 --preprocess
```

To reproduce the paper experiments, see the [USAGE.md](./USAGE.md) page.

_To perform cross-node execution (the WAN setting), you will need to install from source instead._

### From Source

Alternatively, you can also install the code from source.

We assume the AriaNN code has been put at the home directory: `~/ariann`


#### 1. PySyft

Download PySyft from GitHub using the `ryffel/ariaNN` branch and install in editable mode:
```
cd ~
git clone https://github.com/OpenMined/PySyft.git
cd PySyft
git checkout a73b13aa84a8a9ad0923d87ff1b6c8c2facdeaa6
pip install -e .
```
You can already start running the first experiments! Try:
``` 
cd ~/ariann
python main.py --model alexnet --dataset cifar10 --batch_size 128 --preprocess
```
But there are still some extras steps to have the whole setup!

#### 2. PyGrid (for cross-node execution only)
To run experiments across  different nodes, download PyGrid from GitHub.

```
cd ~
git clone https://github.com/OpenMined/PyGrid.git
cd PyGrid
git checkout 1ded5901ce643c1abadd23f54cc84b9c25438f5b
```
Install poetry 
``` 
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```
**Take a fresh tab**, install with poetry (it might take a minute or two):
``` 
cd ~/PyGrid/apps/node
poetry install
poetry env info -p
```
Copy paste the path output, say `<path>`. Now reinstall manually PySyft in the env.
```
source <path>/bin/activate
cd ~/PySyft
pip install -e .
deactivate
```

### Experiments setup

Verify that the paths are correct in ``scripts/launch_***.sh``:
```
cd ~/ariann
nano scripts/launch_alice.sh
nano scripts/launch_bob.sh
nano scripts/launch_crypto_provider.sh
```
Verify that the `HOME` path is correct:
``` 
nano data.py
```

There are 2 datasets that you need to install manually in your `HOME`:
- Hymenoptera using the instructions:
    ```
    cd ~
    wget https://download.pytorch.org/tutorial/hymenoptera_data.zip
    unzip hymenoptera_data.zip
    ```
- Tiny Imagenet, from https://github.com/tjmoon0104/pytorch-tiny-imagenet:
    ```
    cd ~
    git clone https://github.com/tjmoon0104/pytorch-tiny-imagenet.git
    cd pytorch-tiny-imagenet
    pip install opencv-python
    ./run.sh
    ```


    
The working directory is: `cd ~/ariann`
    
    
### Troubleshooting with websockets



When you first launch the workers, you might have a SQL error. That's not important, just re-run the experiment and it will be gone.

Also, we recommend always testing an experiment locally (ie without `--websockets`) before using websockets, to make sure everything runs fine.

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