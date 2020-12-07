import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import KNNGraph, EdgeConv
from src.pointnet2_utils import PointNetSetAbstraction
#https://github.com/yanx27/Pointnet_Pointnet2_pytorch


class FullSplit(nn.Module):
    def __init__(self, type="PointNet", input_dims=3, output_dims=3):
        super(FullSplit, self).__init__()
        if type == "PointNet":
            self.p_extractor = PointNet(input_dims=input_dims)
            self.n_extractor = PointNet(input_dims=input_dims)
        elif type == "PointNet++":
            self.p_extractor = PointNet2(input_dims=input_dims)
            self.n_extractor = PointNet2(input_dims=input_dims)
        elif type == "GNN":
            self.p_extractor = GNN(input_dims=input_dims)
            self.n_extractor = GNN(input_dims=input_dims)
        self.p_predictor = PoseNet()
        self.n_predictor = PoseNet(output_dims=output_dims)

    def forward(self, x):
        feat_p = self.p_extractor(x)
        pred_p = self.p_predictor(feat_p)
        feat_n = self.n_extractor(x)
        pred_n = self.n_predictor(feat_n)
        return pred_p, feat_p, pred_n, feat_n

class PoseNet(nn.Module):
    def __init__(self, input_dims=1024, output_dims=3, dropout_prob=0.5):
        super(PoseNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_dims)
        self.do = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.do(self.fc1(x))),0.2)
        x = F.leaky_relu(self.bn2(self.do(self.fc2(x))),0.2)
        x = self.fc3(x)
        return x

class PointNet(nn.Module):
    def __init__(self, input_dims=3):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dims, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.2)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        feat = x.view(-1, 1024)
        return feat


class PointNet2(nn.Module):
    def __init__(self, input_dims=3, radii=[0.03,0.06]):
        super(PointNet2, self).__init__()
        if input_dims > 3:
            self.extra_channels = True
        else:
            self.extra_channels = False
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=radii[0], nsample=32, in_channel=input_dims, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=radii[1], nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

    def forward(self, x):
        x = x.transpose(2, 1)
        if self.extra_channels:
            norm = x[:, 3:, :]
            x = x[:, :3, :]
        else:
            norm = None
        B, _, _ = x.shape
        l1_x, l1_points = self.sa1(x, norm)
        l2_x, l2_points = self.sa2(l1_x, l1_points)
        _, l3_points = self.sa3(l2_x, l2_points)
        feat = l3_points.view(B, 1024)
        return feat


class GNN(nn.Module):
    def __init__(self, input_dims=3, k=10, feature_dims=[64, 64, 128, 256], emb_dims=[512, 512, 256]):
        super(GNN, self).__init__()
        self.nng = KNNGraph(k)
        self.conv = nn.ModuleList()
        self.num_layers = len(feature_dims)
        for i in range(self.num_layers):
            self.conv.append(EdgeConv(
                feature_dims[i - 1] if i > 0 else input_dims,
                feature_dims[i],
                batch_norm=True))
        self.proj = nn.Linear(sum(feature_dims), emb_dims[0])

    def forward(self, x):
        hs = []
        batch_size, n_points, x_dims = x.shape
        h = x
        for i in range(self.num_layers):
            g = self.nng(h)
            h = h.view(batch_size * n_points, -1)
            h = self.conv[i](g, h)
            h = F.leaky_relu(h, 0.2)
            h = h.view(batch_size, n_points, -1)
            hs.append(h)
        h = torch.cat(hs, 2)
        h = self.proj(h)
        h_max, _ = torch.max(h, 1)
        h_avg = torch.mean(h, 1)
        feat = torch.cat([h_max, h_avg], 1)
        return feat
