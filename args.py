import argparse


args = None


def parse_arguments():


    parser = argparse.ArgumentParser(description='PyTorch Example')
       
 
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')    
    
    parser.add_argument("--workers", default=False, action="store_true",  help="enable workers")
    
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training (default: 128)')
    
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs to train')
    
    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'mobilenet_v1', 'densenet'], help='model to train')
    
    parser.add_argument('--sname', type=str, default='folder',  help='folder save name for runs')
    
    parser.add_argument('--ptarget', type=float, default=0.95,  help='final target pruning ratio')
      
    parser.add_argument('--pthres', type=int, default=0, help='number of minimum required parameters for sparsification (default: 0)')
    
    parser.add_argument('--t1', type=int, default=0, help='start pruning at epoch t1 (default: 0)')
    

    args = parser.parse_args()

    return args



def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()







