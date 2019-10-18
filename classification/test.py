import torch
import torch.nn as nn
from data import get_testloader
from model import LeNet


def test(device=torch.device("cuda:0"), 
          model=None, 
          criterion=nn.CrossEntropyLoss(), 
          testloader=get_testloader(512)):
    if model is None:
        model = Net()
        model.load_state_dict(torch.load('./models/mnist_LeNet.pt'))
        model.eval()
        model.to(device)
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))


if __name__ == '__main__':
    test()