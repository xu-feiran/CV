import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from data import get_trainloader
from deepBar import *

def train(net_name):

    # LeNet-5
    if net_name == 'LeNet':
        save_model = False
        batch_size = 1024
        epoch = 10
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LeNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        trainloader=get_trainloader(batch_size)
    else:
        raise ValueError('No net: %s!'% net_name)
    total = len(trainloader)
    for i in range(0, epoch):
        barBegin(i)
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            barRun(i, total, loss.item())
        barEnd()
    print('Finished Training!')
    if save_model:
        # TODO NetName_DatasetName_Epoch
        torch.save(model.state_dict(), 
                   './models/%s_MNIST_%d.pt'% (net_name, epoch))
        print('Saved model!')
    return model

if __name__ == '__main__':
    train('LeNet')