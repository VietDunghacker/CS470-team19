import torch.nn as nn

!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

random.seed(69)
np.random.seed(69)
torch.manual_seed(69)
torch.cuda.manual_seed(69)

class Network(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0', num_classes = num_classes, advprop = True, include_top=False)
    self.logits = nn.Linear(1280, num_classes)
    self.dropout = nn.Dropout(0.6)
    self.flatten = nn.Flatten()
  def forward(self, x):
    output = self.efficient_net(x)
    output = self.flatten(output)
    output = self.logits(self.dropout(output))
    return output