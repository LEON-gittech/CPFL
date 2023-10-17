import os
# os.chdir('pcode/global_train/') #修改当前工作目录
import sys
sys.path.append('/home/leon/workspace/pFedSD/')
sys.path.append('/home/leon/workspace/pFedSD/pcode/global_train')
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pcode.models.resnet import ResNet_cifar, ResNet_cifar_con
from pcode.datasets.loader.cifar import CIFAR10, CIFAR100
from pcode.datasets import cifar_utils
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import numpy as np

tsne = TSNE()

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
n_epochs = 30

# dataset_loader = CIFAR10
dataset_loader = cifar_utils.CIFAR10Pair
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = DataLoader(
        dataset_loader(
            root="./data", train=True, transform=cifar_utils.train_transform, download=True
        ),
        batch_size=batch_size_train,shuffle=True, drop_last=True
    )

test_loader = DataLoader(
        dataset_loader(
            root="./data", train=False, download=True
        ),
        batch_size=batch_size_test,shuffle=True, drop_last=True
    )

# global train   
network = ResNet_cifar("cifar10", 8).cuda()
# network = ResNet_cifar_con("cifar10", 8).cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
 
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

def train_con(epoch):
    network.train()
    temperature = 0.5
    num_classes = 10
    for batch_idx, (pos1, pos2, data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        pos1, pos2 = pos1.cuda(), pos2.cuda()
        optimizer.zero_grad()
        feature_1, out_1,pred_c_1  = network(pos1)
        feature_2, out_2,pred_c_2 = network(pos2)
        _, pro1 ,output = network(data)
        
        loss = 0
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size_train, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size_train, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        con_loss = 0.25*(- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        #loss 加上 CEL
        pred_c = (pred_c_1+pred_c_2)/2
        tmp = F.one_hot(target.cuda(non_blocking=True),num_classes=num_classes).float()
        cel = F.cross_entropy(pred_c,tmp)
        loss += cel
        loss += con_loss
        
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCEL_Loss: {:.6f}\tCON_Loss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item(),cel.item(),con_loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model_con.pth')
            torch.save(optimizer.state_dict(), './optimizer_con.pth')

def test():
    label = None
    out = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = network(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            if out == []:
                out = output
            else:
                out = torch.cat((out,output),dim=0)
            if label == None:
                label = target
            else:
                label = torch.cat((label,target),dim=0)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return out, label

def test_con():
    label = None
    out = []
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            _,_, output = network(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            if out == []:
                out = output
            else:
                out = torch.cat((out,output),dim=0)
            if label == None:
                label = target
            else:
                label = torch.cat((label,target),dim=0)
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return out, label

# model_state = torch.load("./model_con.pth")
# network.load_state_dict(model_state)
# optimizer.load_state_dict(torch.load("./optimizer_con.pth"))

model_state = torch.load("./model.pth")
network.load_state_dict(model_state)
optimizer.load_state_dict(torch.load("./optimizer.pth"))
# for epoch in range(n_epochs):
#     train_con(epoch)
    
out, label = test()
# out, label = test_con()
out, label = out.cpu(), label.cpu()
out = tsne.fit_transform(out)
# np.save('global_label.py',label)
# np.save('global_tsne.py',out)
np.save('global_label.py',label)
np.save('global_tsne.py',out)

# label = np.load('global_label.py.npy')
# out = np.load('global_tsne.py.npy')
fig = plt.figure()
for i in range(10):
    indices = label == i
	# 标签为i的全部选出来

    x, y = out[indices].T # 这里转置了

	# 画图
    plt.scatter(x, y, label=str(i),s=0.1)
plt.legend()
plt.savefig('tsne.jpg')
plt.show()


