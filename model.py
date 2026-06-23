import torch.nn as nn
import torch.nn.functional as F

from utils import get_device

NUM_CLASSES = 10
DEVICE = get_device()
EPOCHS = 5
LR = 1e-3

class SpuriousCNN(nn.Module):

    def __init__(self, num_classes = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, num_classes)
        self.target_layer = self.conv3

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))  #CAM
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)







