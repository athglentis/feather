import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


import datetime
import os
from pathlib import Path

from sparse_utils import Pruner, get_params

from archs.resnet import resnet
from archs.mobilenet import mobilenet
from archs.densenet import DenseNet3   

from args import args


print(args)


cwd = os.getcwd()
tm = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
writer = SummaryWriter(log_dir=Path(os.path.join(cwd, 'runs', tm+f'_{args.sname}')))


gpu_id = args.gpu
device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() and not args.no_cuda  else torch.device('cpu')



# typical augmentation for CIFAR
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
nclasses = 100

if args.workers:
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8) 
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8) 
else:
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 
    
nbatches = len(train_loader) 

if args.model == 'resnet20':
    model = resnet(depth=20, num_classes=nclasses)
elif args.model == 'mobilenet_v1':
    model = mobilenet(class_num=nclasses)
elif args.model == 'densenet':
    model = DenseNet3(depth=40, num_classes=100, growth_rate=24,
                 reduction=0.5, bottleneck=True, dropRate=0.0)
else:
    raise ValueError('Model not implemented')

model.to(device)

pruner = Pruner(model, device, final_rate=args.ptarget, nbatches=nbatches, epochs=args.epochs, pthres=args.pthres, t1=args.t1)

params, params_nowd = get_params(model)
optimizer = torch.optim.SGD(
    [
        {"params":params_nowd, 'name': 'bnparams', "weight_decay":0},
        {"params":params, 'name': 'parameters'},
    ],
    lr=args.lr, momentum=0.9, weight_decay=args.wd
) 

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs, T_mult=1)

def train(epoch):
    running_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
    
        pruner.update_thresh()

        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        scheduler.step((epoch-1) + batch_idx/nbatches )   
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch} \tLoss: {epoch_loss}')
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    
    pruner.update_thresh(end_of_batch=True)

 

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
     
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n({}) - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
     
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("acc/test",  100. * correct / len(test_loader.dataset), epoch)



for epoch in range(1, args.epochs + 1):     
    
    train(epoch)
    
    pr = pruner.print_sparsity()   
    print(f"prune rate : {pr}" )
    writer.add_scalar("prune_rate", pr, epoch)

    if epoch % 1 == 0:
        test(epoch)


pruner.desparsify()

writer.flush()

