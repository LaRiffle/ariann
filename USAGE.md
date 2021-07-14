# Usage


## Documentation

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

## Reproduce the experiments

### Table 3

To reproduce the results of Table 3, in the LAN setting, you should run the following commands:
```
python main.py --model network1 --dataset mnist --batch_size 128 --preprocess
```
```
python main.py --model network2 --dataset mnist --batch_size 128 --preprocess
```
```
python main.py --model lenet --dataset mnist --batch_size 128 --preprocess
```
```
python main.py --model alexnet --dataset cifar10 --batch_size 128 --preprocess
```
```
python main.py --model alexnet --dataset tiny-imagenet --batch_size 128 --preprocess
```
```
python main.py --model vgg16 --dataset cifar10 --batch_size 64 --preprocess
```
```
python main.py --model vgg16 --dataset tiny-imagenet --batch_size 16 --preprocess
```
```
python main.py --model resnet18 --dataset hymenoptera --batch_size 8 --preprocess
```

_Note that you might need to reduce the `batch_size` if you don't have a RAM of 64GB. Also, to run the WAN setting, you should install from source (i.e. without Docker) and add the `--websockets` option_

### Table 4

To reproduce the results of Table 4, you should run the following commands:

```
python main.py --model network1 --dataset mnist --batch_size 128 --test
```
```
python main.py --model network2 --dataset mnist --batch_size 128 --test
```
```
python main.py --model lenet --dataset mnist --batch_size 128 --test
```
```
python main.py --model alexnet --dataset cifar10 --batch_size 128 --test
```
```
python main.py --model alexnet --dataset tiny-imagenet --batch_size 128 --test
```
```
python main.py --model vgg16 --dataset cifar10 --batch_size 64 --test
```
```
python main.py --model vgg16 --dataset tiny-imagenet --batch_size 16 --test
```
```
python main.py --model resnet18 --dataset hymenoptera --batch_size 8 --test
```
_Add the `--fp_only` option to run the evaluation in fixed precision mode._

### Table 5

To reproduce the results of Table 6, you should run the following commands:

```
python main.py --model network1 --dataset mnist --train --epochs 15 --lr 0.01
```
```
python main.py --model network2 --dataset mnist --train --epochs 10 --lr 0.02 --momentum 0.9
```
```
python main.py --model lenet --dataset mnist --train --epochs 10 --lr 0.02 --momentum 0.9
```

_Add the `--fp_only` option to run the evaluation in fixed precision mode._