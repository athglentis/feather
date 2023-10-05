## Feather: An Elegant Solution to Effective DNN Sparsification

This repository contains the source code accompanying the accepted paper at the 2023 British Machine Vision Conference (BMVC), titled [Feather: An Elegant Solution to Effective DNN Sparsification](https://arxiv.org/abs/2310.02448) by Athanasios Glentis Georgoulakis, George Retsinas and Petros Maragos.


### Overview

**Feather** is a module that enables effective sparsification of neural networks during the standard course of training. The pruning process relies on an enhanced version of the Straight Through Estimator (STE), utilizing a new thresholding operator and a gradient scaling technique, resulting into sparse yet highly accurate models, suitable for compact applications.

Feather is versatile and not bound to a particular pruning framework. For the case of using a backbone based on global magnitude thresholding (i.e. a single threshold selected for all layers) and an incrementally increasing sparsity ratio over the training process, Feather(-Global) results to sparse models with the exact requested sparsity at the end of training which are more accurate than the current state-of-the-art, by a considerable margin. 


### Library Usage

We provide a sketch of how the library that performs DNN pruning using Feather with the Global pruning backbone is used:

```python
import torch
from sparse_utils import Pruner

train_loader = ...
epochs = ...
model = ...
# create a Pruner class instance
pruner = Pruner(model, device=..., final_rate=..., nbatches=..., epochs=...)
optimizer = ...
loss_fn = ...
for epoch in range(epochs):  
    for data, target in train_loader:
        # update the pruning threshold based on the iteration number and the scheduler used
        pruner.update_thresh()    
        output = model(data)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # update the pruning threshold after last step of the optimizer
    pruner.update_thresh(end_of_batch=True)

# finalize sparse model
pruner.desparsify()
```


### Provided files

 - sparse_utils.py: contains an implementation of Feather and the global magnitude pruning framework (Feather-Global) as described in our paper (including all necessary utility functions).

 - main.py:  contains an example use case of Feather-Global on CIFAR-100 (using model architectures ResNet-20, MobileNetV1 and DenseNet40-24 provided in archs directory).

 - args.py:  contains the user-defined arguments regarding training and sparsification choices.

### Tested on PyTorch 1.13 (numpy, torch, torchvision, tensorboard & argparse packages are required)

-------------------------------------------------------------------------

### Main  options: (--gpu, --batch-size, --lr, --wd, --epochs, --model, --ptarget, --sname)

 - gpu: select GPU device id
 - batch-size: batch size for training (default: 128)
 - lr: initial learning rate (default: 0.1, existing scheduler is Cosine Annealing with no warm restarts)
 - wd: weight decay (default: 5e-4)
 - epochs: number of overall epochs (default: 160)
 - model: model architecture to train
 - ptarget: final target pruning ratio
 - sname: folder name for tensorboard log file (final name will be in the form: datetime_sname) 


### Examples

    - python main.py --gpu=0 --wd=5e-4 --epochs=160 --model=resnet20 --ptarget=0.90    --sname='resnet20_ep=160_wd=5e-4_pt=0.90'
	- python main.py --gpu=0 --wd=5e-4 --epochs=160 --model=resnet20 --ptarget=0.99    --sname='resnet20_ep=160_wd=5e-4_pt=0.99'
