from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

class PointNetEncoder(nn.Module):
    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform

    @cached_property
    def embed_dim(self):
        device = next(self.transform.parameters()).device
        x = torch.randn(1, 3, 1, device=device)
        return self(x).shape[-1]

    def forward(self, x):
        """
        Expects (B, 3, N) points, returns (B, embed_dim) features
        """
        return self.transform(x)[0]

def load_pretrained_pn_encoder():
    dl_path = "/tmp/pretrained_pointnet/pointnet.pth"
    import os
    if not os.path.exists(dl_path):
        os.makedirs(os.path.dirname(dl_path), exist_ok=True)
        url = "https://drive.google.com/uc?export=download&id=1FokXSAIIK-uj9QkGmSevIQJKE_-TvK8h"
        import urllib.request
        urllib.request.urlretrieve(url, dl_path)
    model = PointNet()
    model.load_state_dict(torch.load(dl_path, map_location="cpu", weights_only=True))
    return PointNetEncoder(model.transform)

if __name__ == "__main__":
    model = load_pretrained_pn_encoder()
    model.eval()
    point_patches = torch.randn(1, 3, 16, 16, 256)

    gt_features = torch.zeros(1, 16, 16, model.embed_dim)
    with torch.no_grad():
        for i in range(16):
            for j in range(16):
                gt_features[:, i, j, :] = model(point_patches[:, :, i, j, :])

    points = point_patches.permute(0, 2, 3, 1, 4).reshape(-1, 3, 256)
    with torch.no_grad():
        out = model(points)
    patch_features = out.reshape(-1, 16, 16, model.embed_dim)
    print(torch.allclose(gt_features, patch_features, atol=5e-4))
