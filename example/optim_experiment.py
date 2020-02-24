# This is an experiment to compare multiple optimizers over dataset MNIST and CIFAR10.

# from __future__ import print_function
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch.optim as optim
import blurnn.optim as dp_optim

def mnist_dataset_load(setting):
    use_cuda = not setting['no_cuda'] and torch.cuda.is_available()
    kwsetting = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=setting['batch_size'], shuffle=True, **kwsetting)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=setting['test_batch_size'], shuffle=True, **kwsetting)
    return (train_loader, test_loader)
def cifar10_dataset_load(setting):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
        batch_size=setting['batch_size'], shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])),
        batch_size=setting['test_batch_size'], shuffle=False, num_workers=2)

    return (train_loader, test_loader)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

TRAIN_SETTING = {
    'mnist': {
        'dataset_load': mnist_dataset_load,
        'net': Net,
        'loss': {
            'func': F.nll_loss,
            'train_setting': {},
            'test_setting': {
                'reduction': 'sum'
            },
        }
    },
    'cifar10': {
        'dataset_load': cifar10_dataset_load,
        'net': LeNet,
        'loss': {
            'func': nn.CrossEntropyLoss(),
            'train_setting': {},
            'test_setting': {},
        }
    }
}

experiment_results = []

def train(setting, model, device, train_loader, optimizer, epoch, task_name):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = TRAIN_SETTING[setting['dataset']]['loss']['func'](output, target, **TRAIN_SETTING[setting['dataset']]['loss']['train_setting'])
        loss.backward()
        optimizer.step()
        if batch_idx % setting['log_interval'] == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                task_name, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(setting, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += TRAIN_SETTING[setting['dataset']]['loss']['func'](output, target, **TRAIN_SETTING[setting['dataset']]['loss']['test_setting']).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, 100. * correct / len(test_loader.dataset)


def runTrain(setting, task, train_loader, test_loader):
    use_cuda = not setting['no_cuda'] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = TRAIN_SETTING[setting['dataset']]['net']().to(device)

    optimizer = ({
        "Adadelta": optim.Adadelta,
        "Adagrad": optim.Adagrad,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "SGD": optim.SGD,
        "DPSGD": dp_optim.DPSGD,
        "DPAdam": dp_optim.DPAdam,
    })[task['optimizer']](**dict(
        {'params': model.parameters()},
        **task['args']
    ))

    task_result = {
        'name': task['name'],
        'loss': [],
        'accuracy': [],
    }

    scheduler = StepLR(optimizer, step_size=1, gamma=setting['gamma'])
    for epoch in range(1, setting['epochs'] + 1):
        train(setting, model, device, train_loader, optimizer, epoch, task['name'])
        loss, accuracy = test(setting, model, device, test_loader)        
        scheduler.step()

        task_result['loss'].append(loss)
        task_result['accuracy'].append(accuracy)

    experiment_results.append(task_result)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='dp-optimizers Experiments')
    parser.add_argument('--file', '-f', type=str, required=True,
                        help='setting file')

    with open(parser.parse_args().file) as setting_file:
        setting = json.load(setting_file)
        # TODO: validation

    torch.manual_seed(setting['seed'])

    (train_loader, test_loader) = TRAIN_SETTING[setting['dataset']]['dataset_load'](setting)

    for task in setting['tasks']:
        runTrain(setting, task, train_loader, test_loader)

    def formatPrint(field):
        print(f'\n-----{field}------')
        for task_result in experiment_results:
            print(f'{task_result["name"]},{",".join(map(str, task_result[field]))}')

    formatPrint('loss')
    formatPrint('accuracy')

if __name__ == '__main__':
    main()
