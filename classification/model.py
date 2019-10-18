import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet-5
class LeNet(nn.Module):
    r"""LeNet-5 model.

    Reference:
        Gradient-Based Learning Applied to Document Recognition
        https://zhuanlan.zhihu.com/p/30117574
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 120, 5)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x) # C1
        x = F.max_pool2d(x, kernel_size=2, stride=2) # S2
        x = self.conv2(x) # C3
        x = F.max_pool2d(x, kernel_size=2, stride=2) # S4
        x = self.conv3(x) # C5
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)) # F6
        x = self.fc2(x) # The output show be RBF in the paper.
        return x


if __name__ == '__main__':
    net = LeNet()
    print(net)