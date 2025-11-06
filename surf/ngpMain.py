import torch
from dataReader import SurfDataset
from torch.utils.data import DataLoader, Dataset
import network
import torch.optim as optim
from torch_ema import ExponentialMovingAverage
import time
import os
import numpy as np


def ohem_loss(pred, target, keep_num):
    # loss = torch.sum((pred - target)**2, axis=1)
    # thre = torch.mean(loss.detach())
    # pickHem = loss > 0 
    # loss_keep = loss[pickHem]
    # return loss_keep.sum() / torch.sum(pickHem)/3
    return torch.sum((pred - target)**2)
class Trainer(object):
    def __init__(self,opt):
        self.device = opt['device']
        self.dataPath = opt['dataPath']
        self.max_epochs = opt['max_epochs']
        self.batch = opt['batch']
        self.dataload_num_workers = opt['dataload_num_workers']
        self.gridLevelCnt = opt['gridLevelCnt']
        self.eachGridFeatDim = opt['eachGridFeatDim']
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.surfdata = SurfDataset(self.device, self.dataPath)
        self.dataloader = DataLoader(
            self.surfdata, batch_size=self.batch, shuffle=True, num_workers=self.dataload_num_workers)
        print('data load done. dataloader num_workers=', self.dataloader.num_workers)
        self.surfmodel = network.SurfNetwork(
            device, self.surfdata.maxFeatId, self.surfdata.featLevelCnt, self.eachGridFeatDim)
        self.surfmodel.to(self.device)
        self.optimizer = optim.Adam(
            self.surfmodel.parameters(), lr=0.1, weight_decay=5e-4)  # naive adam
        self.scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / 40, 1))
        self.lr_scheduler = self.scheduler(self.optimizer)
        self.ema = ExponentialMovingAverage(
            self.surfmodel.parameters(), decay=0.95)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.criterionSigma = torch.nn.MSELoss(reduction='none')
        pass
    def train(self):
        # if os.path.exists('surf/160_20.save.pt'):
        #     self.surfmodel.load_state_dict(torch.load('surf/160_20.save.pt'))
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            start_time = time.time()
            self.train_one_epoch()
            end_time = time.time()
            print(f"epoch={epoch}, lr={self.lr_scheduler.get_last_lr()},  cost={(end_time - start_time)}s")
            self.lr_scheduler.step()
            self.save_checkpoint()



    def evaluate(self, loader, name=None):
        self.evaluate_one_epoch(loader, name)

    def train_one_epoch(self):
        total_loss = 0
        self.local_step = 0
        self.surfmodel.train()
        for rgb, dirEncode, featId in self.dataloader:
            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()
            B, N, _ = featId.shape
            sigma, colors = self.surfmodel(featId, dirEncode)
            loss0 = self.criterion(rgb, colors).sum()# /B/3
            # loss1 = (-(sigma-0.5)**2).sum()#/B/N
            loss = loss0#+loss1
            loss.backward()
            self.optimizer.step()
            if self.global_step % 2 == 0:
                showLoss0 = loss0.detach()/B/3#loss0.detach().sum()/B
                showLoss1 = 0#loss1.detach()/B/N
                # showLoss2 = loss2.detach()/B
                print(
                    f"batch={B}, loss0={showLoss0}, loss1={showLoss1}")
            # if B != self.batch:
            #     showColors = colors.detach()
            #     showColorsTar = rgb.detach()
            #     print(showColors)
            #     print(showColorsTar)



    def save_checkpoint(self):
        torch.save(self.surfmodel.state_dict(),f'surf/{self.epoch}.pt')


def collectAllPotentialPts(opt, trainDataPath, modelPath, outPath):
    device = torch.device('cpu')
    dataload_num_workers = opt['dataload_num_workers']
    if not os.path.exists(trainDataPath):
        print('not found : ', trainDataPath)
    if not os.path.exists(modelPath):
        print('not found : ', modelPath)
    surfdata = SurfDataset(device, trainDataPath)
    constructionloader = DataLoader(
        surfdata, 64, shuffle=False, num_workers=dataload_num_workers)
    validPos = set()
    with torch.no_grad():
        for rgb, dirEncode, featId in constructionloader:
            validPos.update(featId[..., 0].reshape(-1).numpy())
    pts = np.zeros([len(validPos), 3])
    for i, pos in enumerate(validPos):
        pts[i, 0], pts[i, 1], pts[i, 2] = surfdata.posEncodeToXyz(surfdata.featIdToPosEncode[pos])
    np.savetxt(outPath, pts, delimiter=' ')


def surfPtsConstruct(opt,trainDataPath,modelPath,outPath):

    device = torch.device('cpu')
    dataload_num_workers = opt['dataload_num_workers']
    if not os.path.exists(trainDataPath):
        print('not found : ', trainDataPath)
    if not os.path.exists(modelPath):
        print('not found : ', modelPath)
    surfdata = SurfDataset(device, trainDataPath)
    constructionloader = DataLoader(
        surfdata, 64, shuffle=False, num_workers=dataload_num_workers)
    surfmodel = network.SurfNetwork(
        device, surfdata.maxFeatId, surfdata.featLevelCnt, opt['eachGridFeatDim'])
    surfmodel.load_state_dict(torch.load(modelPath))
    surfmodel.to(device)
    # surfdata.featIdToPosEncode
    validPos=set()
    with torch.no_grad():
        for rgb, dirEncode, featId in constructionloader:
            B, N, _ = featId.shape
            # shape = B,N*featDim,featlevelcnt
            feat = torch.index_select(surfmodel.gridWeight, dim=0, index=featId.reshape(-1)).reshape(B, N*4, -1)
            sigma1234 = feat[..., 0].reshape(B, N, -1)
            sigma1 = torch.clip(sigma1234[..., 0], min=0.0, max=1.0)
            sigma2 = torch.clip(sigma1234[..., 1], min=0.0, max=1.0)
            sigma3 = torch.clip(sigma1234[..., 2], min=0.0, max=1.0)
            sigma4 = torch.clip(sigma1234[..., 3], min=0.0, max=1.0)
            sigma = sigma1*sigma2*sigma3*sigma4
            if torch.max(sigma)>0.5:
                # print(sigma)
                # alpha = 1-sigma
                # transparentBrfore = torch.cumprod(torch.hstack(
                #     [torch.ones([B, 1]), alpha[:, :-1]]), 1)
                # colorWeight = transparentBrfore*sigma
                # print('sigma=', sigma)
                # print('alpha=', alpha)
                # print('transparentBrfore=', transparentBrfore)
                # print('colorWeight=', colorWeight)
                picks = surfdata .potentialGridCnt - \
                    torch.sum(torch.cumsum(sigma > 0.5, axis=1) >= 1, axis=1)
                for b,Pick in enumerate(picks.numpy()):
                    if Pick < surfdata.potentialGridCnt:
                        try:
                            validPos.add(
                                surfdata.featIdToPosEncode[featId[b, Pick, 0].item()])
                        except: 
                            for b  in range(B):
                                print(torch.cumsum(sigma > 0.5, axis=1))
                            exit(0)
 
    pts = np.zeros([len(validPos), 3])
    
    for i,pos in enumerate(validPos):
        pts[i, 0], pts[i, 1], pts[i, 2] = surfdata.posEncodeToXyz(pos)
    np.savetxt(outPath, pts, delimiter=' ')
if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dataload_num_workers =4 
    if device == torch.device('cpu'):
        dataload_num_workers = 4
    opt = {'device': device, 
        'dataPath': 'surf/trainTerm1.dat',
        'max_epochs': 100, 
        'batch': 512000, 
           'dataload_num_workers': dataload_num_workers,
        'gridLevelCnt':6,
        'eachGridFeatDim':4}
    trainer = Trainer(opt)
    trainer.train()
    # surfPtsConstruct(opt, 'surf/trainTerm1.dat','surf/15.pt', 'surf/asd.pts')
    # collectAllPotentialPts(opt, 'surf/trainTerm1.dat','surf/22.pt', 'surf/asd.pts')


