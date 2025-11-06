import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfNetwork(nn.Module):
    def __init__(self,
                 device,
                 maxGridFeatId=0,
                 gridFeatLevelCnt=2,
                 eachGridFeatDim=4,
                 num_layers=2,
                 hidden_dim=4,
                 geo_feat_dim=3,
                 in_dim_dir=16,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 **kwargs,
                 ):
        super().__init__()
        # sigma network
        self.device = device
        self.num_layers = num_layers
        self.eachGridFeatDim = eachGridFeatDim
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.gridFeatLevelCnt = gridFeatLevelCnt
        self.in_dim = gridFeatLevelCnt*eachGridFeatDim
        self.in_dim_dir = in_dim_dir

        self.gridWeight = nn.Parameter(torch.rand(
            maxGridFeatId[-1], eachGridFeatDim), requires_grad=True)#[0,1]

        # sigma_net = []
        # for l in range(num_layers):
        #     if l == 0:
        #         in_dim = self.in_dim
        #     else:
        #         in_dim = hidden_dim
        #     if l == num_layers - 1:
        #         out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
        #     else:
        #         out_dim = hidden_dim
        #     sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))
        # self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color  # 3
        self.hidden_dim_color = hidden_dim_color  # 64
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                # in_dim = self.in_dim_dir + self.geo_feat_dim
                in_dim = self.in_dim_dir + self.gridFeatLevelCnt * self.eachGridFeatDim-self.gridFeatLevelCnt
            else:
                in_dim = hidden_dim_color
            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color
            color_net.append(nn.Linear(in_dim, out_dim, bias=True))
        self.color_net = nn.ModuleList(color_net)

    def forward(self, x, d):  # softmax
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        B, N, _ = x.shape
        feat = torch.index_select(
            self.gridWeight, dim=0, index=x.reshape(-1)).reshape(B, N*self.gridFeatLevelCnt, -1)
        sigma1234 = feat[..., 0].reshape(B, N, -1)
        sigma1 = torch.clip(sigma1234[..., 0], min=0.0, max=1.0).reshape(B, N, 1)
        sigma2 = torch.clip(sigma1234[..., 1], min=0.0, max=1.0).reshape(B, N, 1)
        sigma3 = torch.clip(sigma1234[..., 2], min=0.0, max=1.0).reshape(B, N, 1)
        sigma4 = torch.clip(sigma1234[..., 3],min=0.0, max=1.0).reshape(B, N, 1)
        sigma5 = torch.clip(sigma1234[..., 4],min=0.0, max=1.0).reshape(B, N, 1)
        sigma6 = torch.clip(sigma1234[..., 5],min=0.0, max=1.0).reshape(B, N, 1)
        # sigma1 = 1-torch.exp(-10*sigma1234[..., 0]*sigma1234[..., 0])
        # sigma2 = 1-torch.exp(-10*sigma1234[..., 1]*sigma1234[..., 1])
        # sigma3 = 1-torch.exp(-10*sigma1234[..., 2]*sigma1234[..., 2])
        # sigma4 = 1-torch.exp(-10*sigma1234[..., 3]*sigma1234[..., 3])


  
        # sigma0 = (sigma1*sigma2*sigma3*sigma4+1e-4)
        sigma0 = (sigma1*sigma2*sigma3*sigma4*sigma5*sigma6+1e-4)
        # sigma0 = (sigma1*sigma2+1e-4)

        # sigma = torch.softmax(sigma1*sigma2*sigma3*sigma4,dim=1)
        maxValue,maxIdx =torch.max(sigma0, axis=1,keepdim=True)
        sigma = sigma0/(maxValue)
        geo_feat = feat[..., 1:].reshape(B, N, -1)
        # color
        h = torch.cat([d.unsqueeze(1).tile(1, N, 1), geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        alpha = 1-sigma
        transparentBrfore = torch.cumprod(torch.hstack(
            [torch.ones([B, 1,1]).to(self.device), alpha[:, :-1]]), 2)
        colorWeight0 = transparentBrfore*sigma
        # colorWeight = colorWeight0/torch.sum(colorWeight0,axis=1,keepdim=True)
        colorOut = colorWeight0 *color 
        return sigma, torch.sum(colorOut ,axis=1)
        # return transparentBrfore*sigma, color


    # optimizer utils
    def get_params(self, lr):
        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
            {'params': self.gridWeight, 'lr': lr},
        ]
        return params
