# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
#学習時に使ったのと同じ学習済みモデルをインポート
from torchvision.models import resnet18



# 学習済みモデルに合わせた前処理を追加
transform = transforms.Compose([
    transforms.ToTensor()
])

#　ネットワークの定義
class Net(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.feature = resnet18(pretrained=True)
    #self.feature = mobilenet_v2(pretrained=True)
    #self.feature = mnasnet1_0(pretrained=True)
    self.fc = nn.Linear(1000, 10)
    self.fc2 = nn.Linear(10, 2)


  def forward(self, x):
      h = self.feature(x)
      h = self.fc(h)
      h = F.relu(h)
      h = self.fc2(h)
      return h