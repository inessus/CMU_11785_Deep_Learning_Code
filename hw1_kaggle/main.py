
# coding: utf-8

# In[17]:


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from datetime import datetime

import numpy as np
import os
from WSJ import WSJ



# In[18]:


def to_float_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array)


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


# In[19]:


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_comb, y, k = 10):
        self.k = k
        self.x = x_comb
        self.y = y
	

    def __getitem__(self, idx):
        win = self.x.take(range(idx - self.k, idx + self.k + 1), mode='clip', axis=0).flatten()
        xi = to_float_tensor(win)
        yi = self.y[idx]
        return xi, yi

    def __len__(self):
        return len(self.y)

class MyDataset2(torch.utils.data.Dataset):
    def __init__(self, wsj, k = 10):
        self.k = k
        self.x_list = wsj[0]
        self.y_list = wsj[1] if len(wsj) == 2 else None
        self.idx_map = []
        for i, xs in enumerate(self.x_list):
            for j in range(xs.shape[0]):
                self.idx_map.append((i, j))

    def __getitem__(self, idx):
        i, j = self.idx_map[idx]
        win = self.x_list[i].take(range(j - self.k, j + self.k + 1), mode='clip', axis=0).flatten()
        xi = to_float_tensor(win)
        yi = self.y_list[i][j] if self.y_list is not None else -1
        return xi, yi

    def __len__(self):
        return len(self.idx_map)

class MyDataset3(torch.utils.data.Dataset):
    def __init__(self, wsj, k = 10, lowest=0.1):
        self.k = k
        self.x_list = wsj[0]
        self.y_list = wsj[1] if len(wsj) == 2 else None
        self.idx_map = []
        for i, xs in enumerate(self.x_list):
            for j in range(xs.shape[0]):
                self.idx_map.append((i, j))
        
        self.win_mask = np.concatenate((np.arange(lowest, 1.0, (1 - lowest)/k),
                            np.arange(1.0, lowest, -(1 - lowest)/k),
                            np.array([0.1])))
        self.win_mask = np.repeat(self.win_mask, wsj[0][0].shape[1])
        

    def __getitem__(self, idx):
        i, j = self.idx_map[idx]
        win = self.x_list[i].take(range(j - self.k, j + self.k + 1), mode='clip', axis=0).flatten()
        win *= self.win_mask
        xi = to_float_tensor(win)
        yi = self.y_list[i][j] if self.y_list is not None else -1
        return xi, yi

    def __len__(self):
        return len(self.idx_map)

# In[20]:


def get_data_loaders(args):
    wsj_loader = WSJ(args.data_dir)

    # train_x, train_y = wsj_loader.train
    # test_x, test_y = wsj_loader.dev
    # test_x_comb = np.concatenate(test_x)
    # test_y_comb = np.concatenate(test_y)

    
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),

    #    torch.utils.data.TensorDataset(to_float_tensor(test_x_comb), to_tensor(test_y_comb)),

    #    MyDataset(test_x_comb, test_y_comb, k=args.K),
    #    MyDataset2(wsj_loader.train, k=args.K),
        MyDataset3(wsj_loader.train, k=args.K),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),

    #     torch.utils.data.TensorDataset(to_float_tensor(test_x_comb), to_tensor(test_y_comb)),

    #    MyDataset(test_x_comb, test_y_comb, k=args.K),
    #    MyDataset2(wsj_loader.dev, k=args.K),
        MyDataset3(wsj_loader.dev, k=args.K),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader



# In[26]:


def get_model(k, experiment='e3'):
    in_size = (2 * k + 1) * 40
    sizes = [in_size * n for n in range(10)]
    out_size = 138
    if experiment == 'e1':
        # e1
        return nn.Sequential(
                nn.Linear(in_size, sizes[1]),
                nn.BatchNorm1d(sizes[1]),
                nn.ReLU(),
        
                nn.Linear(sizes[1], sizes[1]),
                nn.BatchNorm1d(sizes[1]),
                nn.ReLU(),
  
                nn.Linear(sizes[1], sizes[1]),
                nn.BatchNorm1d(sizes[1]),
                nn.ReLU(),
        
                nn.Linear(sizes[1], out_size),
            )
    elif experiment == 'e2':
        # e2
        return nn.Sequential(
                nn.Linear(in_size, sizes[2]),
                nn.BatchNorm1d(sizes[2]),
                nn.ReLU(),
        
                nn.Linear(sizes[2], sizes[1]),
                nn.BatchNorm1d(sizes[1]),
                nn.ReLU(),
  
                nn.Linear(sizes[1], sizes[1]),
                nn.BatchNorm1d(sizes[1]),
                nn.ReLU(),

                nn.Linear(sizes[1], sizes[1] // 2),
                nn.BatchNorm1d(sizes[1] // 2),
                nn.ReLU(),
        
                nn.Linear(sizes[1] // 2, out_size),
            )
    # e3
    return nn.Sequential(
            nn.Linear(in_size, sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.ReLU(),
    
            nn.Linear(sizes[1], sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.ReLU(),
  
            nn.Linear(sizes[1], sizes[1]),
            nn.BatchNorm1d(sizes[1]),
            nn.ReLU(),

            nn.Linear(sizes[1], sizes[1] // 2),
            nn.BatchNorm1d(sizes[1] // 2),
            nn.ReLU(),

            nn.Linear(sizes[1] // 2, sizes[1] // 2),
            nn.BatchNorm1d(sizes[1] // 2),
            nn.ReLU(),

            nn.Linear(sizes[1] // 2, sizes[1] // 2),
            nn.BatchNorm1d(sizes[1] // 2),
            nn.ReLU(),
    
            nn.Linear(sizes[1] // 2, out_size),
        )


def train(epoch, model, optimizer, train_loader, args):
    model.train()
    
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} Batch: {} [{}/{} ({:.0f}%, time:{:.2f}s)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), time.time() - t0,
                loss.data[0]))
            t0 = time.time()

def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return "{:.4f}%".format(100. * correct / len(test_loader.dataset))


def main(args):
    print(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_data_loaders(args)
        
    model = get_model(args.K)

    if args.cuda:
        model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(datetime.now())
        train(epoch, model, optimizer, train_loader, args)
        acc_str = test(model, test_loader, args)
        if not os.path.exists(args.weights_dir):
            os.makedirs(args.weights_dir)
        torch.save(model.state_dict(), "{}/{:03d}_{}.w".format(args.weights_dir, epoch, acc_str))


# In[31]:


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--K', type=int, default=10, metavar='N',
                    help='window size')
parser.add_argument('--data-dir', type=str, default='./data/',
                    help='data directory')
parser.add_argument('--weights-dir', type=str, default='./weights/',
                    help='data directory')


# In[ ]:


if __name__ == "__main__":
    main(parser.parse_args())


# In[32]:


#main(parser.parse_args(""))


# In[6]:




