import torch
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F

class MlpNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 120)   # 3072  => 120
        self.fc2 = nn.Linear(120, 84)            # 120   => 84
        self.fc3 = nn.Linear(84, 10)             # 84    => 10

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

# Net = torchvision.models.resnet18(pretrained=True)
# Net = MlpNet()
